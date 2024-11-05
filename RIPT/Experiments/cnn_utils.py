"""
CNN 모델 학습 및 추론에 필요한 유틸리티 함수들을 제공하는 모듈입니다.
"""

import os
import torch
import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import OrderedDict
from torch.utils.data import DataLoader, ConcatDataset, random_split

from Model import cnn_model
from Portfolio import portfolio as pf
from Misc import config as cf
from Data import dgp_config as dcf
from Data.chart_dataset import EquityDataset, TS1DDataset
from Data import equity_data as eqd
from Misc import utilities as ut
import torch.nn as nn

# 데이터 관련 함수들
def get_df_from_dataloader(dataloader: DataLoader) -> pd.DataFrame:
    """데이터로더에서 데이터프레임을 생성합니다."""
    df_columns = ["StockID", "ending_date", "label", "ret_val", "MarketCap"]
    df_dtypes = [object, "datetime64[ns]", np.int, np.float64, np.float64]
    df_list = []
    
    for batch in dataloader:
        batch_df = ut.df_empty(df_columns, df_dtypes)
        batch_df["StockID"] = batch["StockID"]
        batch_df["ending_date"] = pd.to_datetime([str(t) for t in batch["ending_date"]])
        batch_df["label"] = np.nan_to_num(batch["label"].numpy()).reshape(-1)
        batch_df["ret_val"] = np.nan_to_num(batch["ret_val"].numpy()).reshape(-1)
        batch_df["MarketCap"] = np.nan_to_num(batch["MarketCap"].numpy()).reshape(-1)
        df_list.append(batch_df)
        
    df = pd.concat(df_list)
    df.reset_index(drop=True, inplace=True)
    return df

def get_train_validate_dataloaders_dict(
    ws: int,
    pw: int,
    train_freq: str,
    is_years: List[int],
    country: str,
    has_volume_bar: bool,
    has_ma: bool,
    annual_stocks_num: str,
    tstat_threshold: float,
    ohlc_len: int,
    regression_label: Optional[str],
    chart_type: str,
    delayed_ret: int,
    train_size_ratio: float,
    ts1d_model: bool,
    ts_scale: str,
    oos_start_year: int
) -> Dict[str, DataLoader]:
    """학습 및 검증용 데이터로더를 생성합니다."""
    if ts1d_model:
        tv_datasets = {
            year: TS1DDataset(
                ws, pw, train_freq, year,
                country=country,
                remove_tail=(year == oos_start_year - 1),
                ohlc_len=ohlc_len,
                ts_scale=ts_scale,
                regression_label=regression_label,
            )
            for year in is_years
        }
    else:
        tv_datasets = {
            year: EquityDataset(
                ws, pw, train_freq, year,
                country=country,
                has_volume_bar=has_volume_bar,
                has_ma=has_ma,
                annual_stocks_num=annual_stocks_num,
                tstat_threshold=tstat_threshold,
                stockid_filter=None,
                remove_tail=(year == oos_start_year - 1),
                ohlc_len=ohlc_len,
                regression_label=regression_label,
                chart_type=chart_type,
                delayed_ret=delayed_ret,
            )
            for year in is_years
        }
    
    tv_dataset = ConcatDataset([tv_datasets[year] for year in is_years])
    train_size = int(len(tv_dataset) * train_size_ratio)
    validate_size = len(tv_dataset) - train_size
    
    train_dataset, validate_dataset = random_split(tv_dataset, [train_size, validate_size])
    
    batch_size = cf.BATCH_SIZE * max(torch.cuda.device_count(), 1)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cf.NUM_WORKERS,
    )
    validate_dataloader = DataLoader(
        validate_dataset,
        batch_size=batch_size,
        num_workers=cf.NUM_WORKERS
    )
    
    return {"train": train_dataloader, "validate": validate_dataloader}

# 모델 관련 함수들
def load_model_state_dict_from_save_path(model_save_path: str, device: torch.device) -> OrderedDict:
    """저장된 모델의 state dict를 로드합니다."""
    print(f"Loading model state dict from {model_save_path}")
    checkpoint = torch.load(model_save_path, map_location=device)
    model_state_dict = checkpoint["model_state_dict"]
    
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
        
    return new_state_dict

# 메트릭 관련 함수들
def save_training_metrics(model_dir: str, val_df: pd.DataFrame, train_df: Optional[pd.DataFrame], ensem: int) -> None:
    """학습 메트릭을 저장합니다."""
    metrics = {
        "validation": val_df.to_dict(),
        "train": train_df.to_dict() if train_df is not None else None
    }
    
    yaml_path = os.path.join(model_dir, f"training_metrics_ensem{ensem}.yaml")
    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(metrics, yaml_file, default_flow_style=False)
    
    print(f"학습 메트릭이 {yaml_path}에 저장되었습니다.")

def calculate_oos_up_prob(ensem_res: pd.DataFrame) -> pd.Series:
    """OOS 상승 확률을 계산합니다."""
    return pd.Series({
        "Sample Number": len(ensem_res),
        "True Up Pct": np.sum(ensem_res.period_ret > 0.0) / len(ensem_res),
        "Pred Up Pct": np.sum(ensem_res.up_prob > 0.5) / len(ensem_res),
        "Mean Up Prob": ensem_res.up_prob.mean()
    })

# 경로 관련 함수들
def get_portfolio_dir(
    country: str,
    model_name: str,
    ws: int,
    pw: int,
    ensem: int,
    oos_years: List[int],
    pf_freq: str,
    delayed_ret: int
) -> str:
    """포트폴리오 디렉토리 경로를 생성합니다."""
    name_parts = [country]
    
    if model_name not in cf.BENCHMARK_MODEL_NAME_DICT.values():
        name_parts.append(model_name)
    
    name_parts.extend([
        f"{ws}d{pw}p",
        f"e{ensem}",
        f"{oos_years[0]}-{oos_years[-1]}"
    ])
    
    if pf_freq != dcf.FREQ_DICT[pw]:
        name_parts.append(f"{pf_freq}")
    
    if delayed_ret != 0:
        name_parts.append(f"d{delayed_ret}")
    
    name = "_".join(name_parts)
    return cf.get_dir(os.path.join(cf.PORTFOLIO_DIR, name))

def save_exp_params_to_yaml(
    model_dir: str,
    params: Dict[str, Any]
) -> None:
    """실험 파라미터를 YAML 파일로 저장합니다."""
    yaml_path = os.path.join(model_dir, "exp_params.yaml")
    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(params, yaml_file, default_flow_style=False)
    
    print(f"실험 파라미터가 {yaml_path}에 저장되었습니다.")

# 데이터셋 관련 함수들
def get_dataloader_for_year(
    ws: int,
    pw: int,
    freq: str,
    year: int,
    country: str,
    has_volume_bar: bool,
    has_ma: bool,
    annual_stocks_num: str,
    tstat_threshold: float,
    ohlc_len: int,
    regression_label: Optional[str],
    chart_type: str,
    delayed_ret: int,
    ts1d_model: bool,
    ts_scale: str,
    oos_start_year: int,
    remove_tail: bool = False
) -> DataLoader:
    """연도별 데이터로더를 생성합니다."""
    if ts1d_model:
        dataset = TS1DDataset(
            ws, pw, freq, year,
            country=country,
            remove_tail=remove_tail,
            ohlc_len=ohlc_len,
            ts_scale=ts_scale,
            regression_label=regression_label,
            )
    else:
        dataset = EquityDataset(
            ws, pw, freq, year,
            country=country,
            has_volume_bar=has_volume_bar,
            has_ma=has_ma,
            annual_stocks_num=annual_stocks_num,
            tstat_threshold=tstat_threshold,
            stockid_filter=None,
            remove_tail=remove_tail,
            ohlc_len=ohlc_len,
            regression_label=regression_label,
            chart_type=chart_type,
            delayed_ret=delayed_ret,
        )
    
    batch_size = cf.BATCH_SIZE * max(torch.cuda.device_count(), 1)
    return DataLoader(dataset, batch_size=batch_size)

# 앙상블 관련 함수들
def load_ensemble_res(
    year: Optional[Union[int, List[int]]],
    ensem_res_dir: str,
    ensem: int,
    ws: int,
    pw: int,
    ohlc_len: int,
    freq: str,
    country: str,
    multiindex: bool = False
) -> pd.DataFrame:
    """앙상블 결과를 로드합니다."""
    year_list = ([] if year is None else [year] if isinstance(year, int) else year)
    df_list = []
    
    for y in year_list:
        ohlc_str = f"{ohlc_len}ohlc" if ohlc_len != ws else ""
        print(f"Loading {ws}d{pw}p{ohlc_str} ensem results for year {y} with freq {freq}")
        
        freq_surfix = f"_{freq}" if freq != dcf.FREQ_DICT[pw] else ""
        ensem_res_path = os.path.join(ensem_res_dir, f"ensem{ensem}_res_{y}{freq_surfix}.csv")
        
        if os.path.exists(ensem_res_path):
            print(f"Loading from {ensem_res_path}")
            df = pd.read_csv(
                ensem_res_path,
                parse_dates=["ending_date"],
                index_col=0,
                engine="python",
            )
            df.StockID = df.StockID.astype(str)
            df_list.append(df)
    
    whole_ensemble_res = pd.concat(df_list, ignore_index=True)
    whole_ensemble_res.rename(columns={"ending_date": "Date"}, inplace=True)
    whole_ensemble_res.set_index(["Date", "StockID"], inplace=True)
    
    if country == "USA":
        whole_ensemble_res = whole_ensemble_res[["up_prob", "MarketCap"]]
        
    if not multiindex:
        whole_ensemble_res.reset_index(inplace=True, drop=False)
        
    whole_ensemble_res.dropna(inplace=True)
    return whole_ensemble_res

def load_ensemble_res_with_period_ret(
    year: Optional[Union[int, List[int]]],
    freq: str,
    country: str,
    ensem_res_dir: str,
    ensem: int,
    ws: int,
    pw: int,
    ohlc_len: int
) -> pd.DataFrame:
    """기간 수익률이 포함된 앙상블 결과를 로드합니다."""
    ensem_res = load_ensemble_res(
        year=year,
        multiindex=True,
        freq=freq,
        ensem_res_dir=ensem_res_dir,
        ensem=ensem,
        ws=ws,
        pw=pw,
        ohlc_len=ohlc_len,
        country=country
    )
    
    period_ret = eqd.get_period_ret(freq, country=country)
    print(f"Loading ensem res with {freq} return of no delay")
    ensem_res["period_ret"] = period_ret[f"next_{freq}_ret"]
    ensem_res.dropna(inplace=True)
    
    return ensem_res

# 메트릭 관련 함수들
def calculate_true_up_prob_from_dataloader(dataloader: DataLoader) -> Tuple[int, float]:
    """데이터로더에서 실제 상승 확률을 계산합니다."""
    df = get_df_from_dataloader(dataloader)
    assert np.sum(df.label == 1) + np.sum(df.label == 0) == len(df)
    return len(df), np.sum(df.label == 1) / len(df)

def df_true_up_label(
    year_list: List[int],
    datatype: str,
    ensem_res_dir: str,
    ensem: int,
    ws: int,
    pw: int,
    ohlc_len: int,
    freq: str,
    country: str
) -> pd.DataFrame:
    """연도별 실제 상승 레이블 통계를 계산합니다."""
    df = pd.DataFrame(
        index=year_list,
        columns=[
            "Sample Number",
            "True Up Pct",
            "Accy",
            "Pred Up Pct",
            "Mean Up Prob",
            "Accy (Pred Up)",
            "Accy (Pred Down)",
        ],
    )
    
    for y in year_list:
        try:
            ensem_res = load_ensemble_res_with_period_ret(
                year=y,
                freq=freq,
                country=country,
                ensem_res_dir=ensem_res_dir,
                ensem=ensem,
                ws=ws,
                pw=pw,
                ohlc_len=ohlc_len
            )
        except FileNotFoundError:
            continue

        print(f"{np.sum(np.isnan(ensem_res.period_ret))}/{len(ensem_res)} of ret_val is Nan")
        label = np.where(ensem_res.period_ret > 0, 1, 0)
        pred = np.where(ensem_res.up_prob > 0.5, 1, 0)
        
        df.loc[y, "Sample Number"] = len(ensem_res)
        df.loc[y, "True Up Pct"] = np.sum(ensem_res.period_ret > 0.0) / len(ensem_res)
        df.loc[y, "Accy"] = np.sum(label == pred) / len(label)
        df.loc[y, "Pred Up Pct"] = np.sum(ensem_res.up_prob > 0.5) / len(ensem_res)
        df.loc[y, "Mean Up Prob"] = ensem_res.up_prob.mean()
        df.loc[y, "Accy (Pred Up)"] = np.sum(label[pred == 1] == pred[pred == 1]) / len(pred[pred == 1])
        df.loc[y, "Accy (Pred Down)"] = np.sum(label[pred == 0] == pred[pred == 0]) / len(pred[pred == 0])
        
    df.loc[f"{datatype} Mean"] = df.mean(axis=0)
    df = df.astype(float).round(2)
    df["Sample Number"] = df["Sample Number"].astype(int)
    
    return df

# 포트폴리오 관련 함수들
def load_portfolio_obj(
    whole_ensemble_res: Optional[pd.DataFrame],
    pf_freq: str,
    pf_dir: str,
    country: str,
    delay_list: List[int],
    model_name: str,
    load_signal: bool = True,
    custom_ret: Optional[str] = None,
    transaction_cost: bool = False,
    start_year: int = cf.OOS_YEARS[0],
    end_year: int = cf.OOS_YEARS[-1]
) -> pf.PortfolioManager:
    """포트폴리오 매니저 객체를 생성합니다."""
    pf_obj = pf.PortfolioManager(
        signal_df=whole_ensemble_res,
        freq=pf_freq,
        portfolio_dir=pf_dir,
        start_year=start_year,
        end_year=end_year,
        country=country,
        delay_list=delay_list,
        load_signal=load_signal,
        custom_ret=custom_ret,
        transaction_cost=transaction_cost,
        model_name=model_name
    )
    return pf_obj

def get_pf_res_path(
    pf_dir: str,
    weight_type: str,
    delay: int = 0,
    cut: int = 10,
    file_ext: str = "csv",
    rank_weight: Optional[str] = None
) -> str:
    """포트폴리오 결과 파일 경로를 생성합니다."""
    assert weight_type in ["vw", "ew"]
    delay_prefix = "" if delay == 0 else f"{delay}d_delay_"
    cut_surfix = "" if cut == 10 else f"_{cut}cut"
    rank_weight_surfix = "" if rank_weight is None else f"_rank_{rank_weight}"
    pf_name = f"{delay_prefix}{weight_type}_{cut_surfix}{rank_weight_surfix}"
    return os.path.join(pf_dir, f"{pf_name}.{file_ext}")

def get_pf_data(
    pf_dir: str,
    weight_type: str,
    value_filter: int = 100,
    delay: int = 0,
    cut: int = 10,
    rank_weight: Optional[str] = None
) -> pd.DataFrame:
    """포트폴리오 데이터를 로드합니다."""
    delay_prefix = "" if delay == 0 else f"{delay}d_delay_"
    cut_surfix = "" if cut == 10 else f"_{cut}cut"
    rank_weight_surfix = "" if rank_weight is None else f"_rank_{rank_weight}"
    pf_name = f"{delay_prefix}{weight_type}_{value_filter}{cut_surfix}{rank_weight_surfix}"
    pf_data_path = os.path.join(pf_dir, "pf_data", f"pf_data_{pf_name}.csv")
    print(f"Loading portfolio data from {pf_data_path}")
    return pd.read_csv(pf_data_path, index_col=0, parse_dates=True)

# OOS 메트릭 관련 함수들
def calculate_oos_metrics(
    ensem_res: pd.DataFrame,
    regression_label: Optional[str]
) -> Dict[str, float]:
    """OOS 메트릭을 계산합니다."""
    
    ret_name = "period_ret"
    print(ensem_res.columns)
    print(ensem_res.head())
    
    pred_prob = ensem_res.up_prob.to_numpy()
    label = np.where(ensem_res[ret_name].to_numpy() > 0, 1, 0)
    
    if regression_label is not None:
        pred_prob += 0.5
        
    oos_metrics = ut.calculate_test_log(pred_prob, label)
    
    # 상관계수 계산
    rank_corr = ensem_res.groupby("Date").apply(
        lambda df: df["up_prob"].rank(method="average", ascending=False).corr(
            df[ret_name].rank(method="average", ascending=False),
            method="spearman"
        )
    )
    pearson_corr = ensem_res.groupby("Date").apply(
        lambda df: pd.Series(df["up_prob"]).corr(df[ret_name], method="pearson")
    )
    
    oos_metrics["Spearman"] = rank_corr.mean()
    oos_metrics["Pearson"] = pearson_corr.mean()
    
    return oos_metrics

def save_oos_metrics(metrics: Dict[str, float], metrics_path: str) -> None:
    """OOS 메트릭을 저장합니다."""
    ut.save_pkl_obj(metrics, metrics_path)
    print(f"OOS metrics saved to {metrics_path}")

def load_oos_metrics(metrics_path: str) -> Dict[str, float]:
    """저장된 OOS 메트릭을 로드합니다."""
    if os.path.exists(metrics_path):
        print(f"Loading OOS metrics from {metrics_path}")
        return ut.load_pkl_obj(metrics_path)
    return None

# 실험 파라미터 관련 함수들
def get_exp_params(
    ws: int,
    pw: int,
    model_obj: cnn_model.Model,
    train_freq: str,
    drop_prob: float,
    has_ma: bool,
    has_volume_bar: bool,
    delayed_ret: int,
    learning_rate: float,
    weight_decay: float,
    loss_name: str,
    annual_stocks_num: str,
    tstat_threshold: float,
    ohlc_len: int,
    train_size_ratio: float,
    ts_scale: str,
    chart_type: str,
    country: str,
    transfer_learning: Optional[str],
    margin: Optional[float] = None
) -> Dict[str, Any]:
    """실험 파라미터 딕셔너리를 생성합니다."""
    params = {
        "window_size": ws,
        "prediction_window": pw,
        "learning_rate": learning_rate,
        "dropout_prob": drop_prob,
        "has_moving_average": has_ma,
        "has_volume_bar": has_volume_bar,
        "train_freq": train_freq,
        "delayed_ret": delayed_ret,
        "batch_norm": model_obj.batch_norm,
        "xavier_init": model_obj.xavier,
        "leaky_relu": model_obj.lrelu,
        "weight_decay": weight_decay,
        "loss_function": loss_name,
        "annual_stocks_num": annual_stocks_num,
        "tstat_threshold": tstat_threshold,
        "ohlc_len": ohlc_len,
        "train_size_ratio": train_size_ratio,
        "regression_label": model_obj.regression_label,
        "ts_scale": ts_scale,
        "chart_type": chart_type,
        "country": country,
        "transfer_learning": transfer_learning
    }
    
    if loss_name == "multimarginloss" and margin is not None:
        params["margin"] = margin
        
    return params

# 메모리 관리 함수들
def release_dataloader_memory(dataloaders_dict: Dict[str, DataLoader], model: nn.Module) -> None:
    """데이터로더와 모델의 메모리를 해제합니다."""
    for key in list(dataloaders_dict.keys()):
        dataloaders_dict[key] = None
    del model
    torch.cuda.empty_cache()

# 경로 생성 함수
def get_model_checkpoint_path(model_dir: str, model_num: int, country: str, tl: Optional[str]) -> str:
    """모델 체크포인트 경로를 반환합니다."""
    if country != "USA" and tl == "usa":
        print(f"Using pretrained USA model for {country}-usa")
        model_dir = model_dir.replace(f"-{country}-usa", "")
    return os.path.join(model_dir, f"checkpoint{model_num}.pth.tar")

# 데이터 로딩 함수
def load_ensemble_model_paths(ensem: int, model_dir: str, country: str, tl: Optional[str]) -> List[str]:
    """앙상블 모델 경로 리스트를 반환합니다."""
    return [get_model_checkpoint_path(model_dir, i, country, tl) for i in range(ensem)]