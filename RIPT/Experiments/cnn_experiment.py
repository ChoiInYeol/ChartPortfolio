"""
CNN 모델 실험을 위한 메인 모듈입니다.
"""

import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from typing import List, Optional

from Model import cnn_model
from Misc import config as cf
from Data import dgp_config as dcf
from Data import equity_data as eqd

from .cnn_utils import (
    get_portfolio_dir,
    get_train_validate_dataloaders_dict,
    save_training_metrics,
    load_model_state_dict_from_save_path,
    get_dataloader_for_year,
    load_ensemble_res,
    calculate_oos_metrics,
    save_oos_metrics,
    load_portfolio_obj,
    release_dataloader_memory,
    get_model_checkpoint_path,
    load_ensemble_model_paths
)
from .cnn_train import CNNTrainer
from .cnn_inference import CNNInference

class Experiment:
    """CNN 모델 실험을 위한 클래스입니다."""
    
    def __init__(
        self,
        ws: int,
        pw: int,
        model_obj: cnn_model.Model,
        train_freq: str,
        ensem: int = 5,
        lr: float = 1e-5,
        drop_prob: float = 0.50,
        max_epoch: int = 50,
        enable_tqdm: bool = False,
        early_stop: bool = True,
        has_ma: bool = True,
        has_volume_bar: bool = True,
        is_years: List[int] = cf.IS_YEARS,
        oos_years: List[int] = cf.OOS_YEARS,
        country: str = "USA",
        transfer_learning: Optional[str] = None,
        annual_stocks_num: str = "all",
        tstat_threshold: float = 0,
        ohlc_len: Optional[int] = None,
        pf_freq: Optional[str] = None,
        tensorboard: bool = False,
        weight_decay: float = 0,
        loss_name: str = "cross_entropy",
        margin: float = 1,
        train_size_ratio: float = 0.7,
        ts_scale: str = "image_scale",
        chart_type: str = "bar",
        delayed_ret: int = 0,
    ):
        """
        Args:
            ws: 윈도우 사이즈
            pw: 예측 윈도우 사이즈
            model_obj: CNN 모델 객체
            train_freq: 학습 주기
            ensem: 앙상블 수
            lr: 학습률
            drop_prob: 드롭아웃 확률
            max_epoch: 최대 에폭 수
            enable_tqdm: tqdm 사용 여부
            early_stop: 조기 종료 여부
            has_ma: 이동평균선 포함 여부
            has_volume_bar: 거래량 바 포함 여부
            is_years: 학습 연도 리스트
            oos_years: OOS 연도 리스트
            country: 국가 코드
            transfer_learning: 전이학습 방식
            annual_stocks_num: 연간 주식 수
            tstat_threshold: t-통계량 임계값
            ohlc_len: OHLC 길이
            pf_freq: 포트폴리오 주기
            tensorboard: 텐서보드 사용 여부
            weight_decay: 가중치 감쇠
            loss_name: 손실 함수 이름
            margin: 마진 값
            train_size_ratio: 학습 데이터 비율
            ts_scale: 시계열 스케일링 방법
            chart_type: 차트 타입
            delayed_ret: 지연된 수익률 기간
        """
        # 기본 설정
        self.ws = ws
        self.pw = pw
        self.model_obj = model_obj
        self.train_freq = train_freq
        self.ensem = ensem
        self.lr = lr
        self.drop_prob = drop_prob
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_epoch = max_epoch
        self.enable_tqdm = enable_tqdm
        self.early_stop = early_stop
        
        # 데이터 관련 설정
        self.has_ma = has_ma
        self.has_volume_bar = has_volume_bar
        self.is_years = is_years
        self.oos_years = oos_years
        self.country = country
        self.tl = transfer_learning
        self.annual_stocks_num = annual_stocks_num
        self.tstat_threshold = tstat_threshold
        self.ohlc_len = ohlc_len if ohlc_len is not None else ws
        self.ts_scale = ts_scale
        self.chart_type = chart_type
        self.delayed_ret = delayed_ret
        
        # 학습 관련 설정
        self.weight_decay = weight_decay
        self.loss_name = loss_name
        self.margin = margin
        self.train_size_ratio = train_size_ratio
        self.tensorboard = tensorboard
        
        # 포트폴리오 관련 설정
        self.pf_freq = pf_freq if pf_freq is not None else dcf.FREQ_DICT[pw]
        
        # 경로 설정
        self.exp_name = f"{self.ws}D{self.pw}P"
        
        self.model_dir = os.path.join(cf.WORK_DIR, self.model_obj.name, self.exp_name)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 앙상블 결과 디렉토리 설정
        self.ensem_res_dir = os.path.join(self.model_dir, "ensem_res")
        os.makedirs(self.ensem_res_dir, exist_ok=True)
        
        self.oos_metrics_path = os.path.join(self.ensem_res_dir, "oos_metrics_no_delay.pkl")
        
        self.pf_dir = get_portfolio_dir(
            country=self.country,
            model_name=self.model_obj.name,
            ws=self.ws,
            pw=self.pw,
            ensem=self.ensem,
            oos_years=self.oos_years,
            pf_freq=self.pf_freq,
            delayed_ret=self.delayed_ret
        )
        
        # 학습기 및 추론기 초기화
        self.trainer = CNNTrainer(
            model=self.model_obj.init_model(),
            device=self.device,
            loss_name=self.loss_name
        )
        
        self.inferencer = CNNInference(
            model=self.model_obj.init_model(),
            device=self.device,
            regression_label=self.model_obj.regression_label
        )

    def train_empirical_ensem_model(self, ensem_range=None, pretrained=True):
        """앙상블 모델을 학습합니다."""
        if ensem_range is None:
            ensem_range = range(self.ensem)
            
        val_df = pd.DataFrame(columns=["MCC", "loss", "accy", "diff", "epoch"])
        train_df = pd.DataFrame(columns=["MCC", "loss", "accy", "diff", "epoch"])
        
        for model_num in ensem_range:
            print(f"Start Training Ensem Number {model_num}", end=" ")
            model_save_path = self._get_model_checkpoint_path(model_num)
            
            if os.path.exists(model_save_path) and pretrained:
                print(f"Found pretrained model {model_save_path}")
                validate_metrics = torch.load(model_save_path, map_location=self.device, weights_only=True)
            else:
                dataloaders_dict = get_train_validate_dataloaders_dict(
                    ws=self.ws,
                    pw=self.pw,
                    train_freq=self.train_freq,
                    is_years=self.is_years,
                    country=self.country,
                    has_volume_bar=self.has_volume_bar,
                    has_ma=self.has_ma,
                    annual_stocks_num=self.annual_stocks_num,
                    tstat_threshold=self.tstat_threshold,
                    ohlc_len=self.ohlc_len,
                    regression_label=self.model_obj.regression_label,
                    chart_type=self.chart_type,
                    delayed_ret=self.delayed_ret,
                    train_size_ratio=self.train_size_ratio,
                    ts1d_model=self.model_obj.ts1d_model,
                    ts_scale=self.ts_scale,
                    oos_start_year=self.oos_years[0]
                )
                
                optimizer = optim.Adam(
                    self.trainer.model.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay
                )
                
                train_metrics, validate_metrics, _ = self.trainer.train_single_model(
                    dataloaders_dict=dataloaders_dict,
                    model_save_path=model_save_path,
                    optimizer=optimizer,
                    max_epoch=self.max_epoch,
                    early_stop=self.early_stop,
                    enable_tqdm=self.enable_tqdm
                )
                
                for column in train_metrics.keys():
                    train_df.loc[model_num, column] = train_metrics[column]

            for column in validate_metrics.keys():
                if column == "model_state_dict":
                    continue
                val_df.loc[model_num, column] = validate_metrics[column]

        val_df = val_df.astype(np.float64).round(3)
        val_df.loc["Mean"] = val_df.mean().round(3)
        
        save_training_metrics(self.model_dir, val_df, train_df, self.ensem)

    def calculate_portfolio(
        self,
        load_saved_data: bool = True,
        delay_list: List[int] = [0],
        is_ensem_res: bool = True,
        cut: int = 10
    ) -> None:
        """
        포트폴리오를 계산하고 결과를 저장합니다.

        Args:
            load_saved_data: 저장된 데이터 사용 여부
            delay_list: 지연 기간 리스트
            is_ensem_res: 앙상블 결과 사용 여부
            cut: 포트폴리오 분할 수
        """
        # 앙상블 결과 생성
        ensem_res_year_list = (
            list(self.is_years) + list(self.oos_years)
            if is_ensem_res
            else list(self.oos_years)
        )
        
        # 각 연도별 앙상블 결과 생성
        for year in ensem_res_year_list:
            freq_surfix = f"_{self.pf_freq}" if self.pf_freq != dcf.FREQ_DICT[self.pw] else ""
            ensem_res_path = os.path.join(
                self.ensem_res_dir,
                f"ensem{self.ensem}_res_{year}{freq_surfix}.csv"
            )
            
            if os.path.exists(ensem_res_path) and load_saved_data:
                print(f"Found {ensem_res_path}")
                continue
                
            print(f"Generating {self.ws}d{self.pw}p ensem results for year {year}")
            
            # 데이터로더 생성
            year_dataloader = get_dataloader_for_year(
                ws=self.ws,
                pw=self.pw,
                freq=self.pf_freq,
                year=year,
                country=self.country,
                has_volume_bar=self.has_volume_bar,
                has_ma=self.has_ma,
                annual_stocks_num=self.annual_stocks_num,
                tstat_threshold=self.tstat_threshold,
                ohlc_len=self.ohlc_len,
                regression_label=self.model_obj.regression_label,
                chart_type=self.chart_type,
                delayed_ret=self.delayed_ret,
                ts1d_model=self.model_obj.ts1d_model,
                ts_scale=self.ts_scale,
                oos_start_year=self.oos_years[0],
                remove_tail=(year == self.oos_years[0] - 1) and self.pf_freq != "year"
            )
            
            # 앙상블 모델 로드
            model_paths = load_ensemble_model_paths(self.ensem, self.model_dir, self.country, self.tl)
            model_list = self.inferencer.load_ensemble_model(model_paths, self.ensem)
            
            if model_list is None:
                print(f"Skipping year {year} due to missing models")
                continue
                
            # 앙상블 결과 생성
            df = self.inferencer.ensemble_results(model_list, year_dataloader)
            df.to_csv(ensem_res_path)
            
            # 메모리 정리
            release_dataloader_memory({"train": year_dataloader}, model_list[0])
            del model_list
            torch.cuda.empty_cache()

        # 앙상블 결과 로드
        whole_ensemble_res = load_ensemble_res(
            year=self.oos_years,
            ensem_res_dir=self.ensem_res_dir,
            ensem=self.ensem,
            ws=self.ws,
            pw=self.pw,
            ohlc_len=self.ohlc_len,
            freq=self.pf_freq,
            country=self.country,
            multiindex=True
        )
        
        # 앙상블 결과에 period_ret 추가
        period_ret = eqd.get_period_ret(period=self.pf_freq, country=self.country)
        whole_ensemble_res["period_ret"] = period_ret[f"next_{self.pf_freq}_ret"]
        whole_ensemble_res.dropna(inplace=True)
        
        # OOS 메트릭 계산 및 저장
        oos_metrics = calculate_oos_metrics(
            whole_ensemble_res,
            self.model_obj.regression_label
        )
        save_oos_metrics(oos_metrics, self.oos_metrics_path)
        print("OOS Metrics:", oos_metrics)

        # 포트폴리오 매니저 초기화 및 포트폴리오 생성
        if self.delayed_ret != 0:
            delay_list = delay_list + [self.delayed_ret]
            
        pf_obj = load_portfolio_obj(
            whole_ensemble_res=whole_ensemble_res,
            pf_freq=self.pf_freq,
            pf_dir=self.pf_dir,
            country=self.country,
            delay_list=delay_list,
            model_name=self.model_obj.name,
            start_year=self.oos_years[0],
            end_year=self.oos_years[-1]
        )

        # 포트폴리오 생성 및 결과 저장
        for delay in delay_list:
            pf_obj.generate_portfolio(delay=delay, cut=cut)
        
        # 포트폴리오 플롯 생성
        pf_obj.make_portfolio_plot(
            portfolio_ret=None,  # 자동으로 계산됨
            cut=cut,
            weight_type="ew"  # 동일가중 포트폴리오
        )

    def _get_model_checkpoint_path(self, model_num: int) -> str:
        """모델 체크포인트 경로를 반환합니다."""
        return get_model_checkpoint_path(self.model_dir, model_num, self.country, self.tl)
