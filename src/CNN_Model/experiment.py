import os
import sys

# 프로젝트 루트 경로를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.CNN_Model.Experiments.cnn_experiment import Experiment
from src.CNN_Model.Model import cnn_model
from src.CNN_Model.Misc import config as cf
from src.CNN_Model.Data import dgp_config as dcf
from src.CNN_Model.Experiments.cnn_utils import (
    get_portfolio_dir,
    save_exp_params_to_yaml,
    save_training_metrics,
    calculate_oos_up_prob,
    load_ensemble_res,
    load_ensemble_res_with_period_ret,
    save_oos_metrics
)

from typing import Optional
import torch
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
def set_device(gpu_ids: str) -> torch.device:
    """
    GPU ID를 설정하고 디바이스를 반환합니다.

    Args:
        gpu_ids (str): 사용할 GPU ID 문자열 (예: "0,1,2,3")

    Returns:
        torch.device: 설정된 디바이스
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    return device

def create_model_object(
    ws: int,
    ts1d_model: bool = False,
    layer_number: Optional[int] = None,
    inplanes: int = cf.TRUE_DATA_CNN_INPLANES,
    drop_prob: float = 0.50,
    batch_norm: bool = True,
    xavier: bool = True,
    lrelu: bool = True,
    regression_label: Optional[str] = None
) -> cnn_model.Model:
    """
    CNN 모델 객체를 생성합니다.

    Args:
        ws: 윈도우 사이즈
        ts1d_model: 1D CNN 모델 여부
        layer_number: 레이어 수 (None이면 기본값 사용)
        inplanes: 초기 채널 수
        drop_prob: 드롭아웃 확률
        batch_norm: 배치 정규화 사용 여부
        xavier: Xavier 초기화 사용 여부
        lrelu: LeakyReLU 사용 여부
        regression_label: 회귀 레이블 (옵션)

    Returns:
        Model 객체
    """
    # 기본 레이어 수 설정
    if layer_number is None:
        if ts1d_model:
            layer_number = cf.TS1D_LAYERNUM_DICT[ws]
        else:
            layer_number = cf.BENCHMARK_MODEL_LAYERNUM_DICT[ws]
    
    # 필터 크기, 스트라이드, 팽창, 맥스풀링 설정
    if ts1d_model:
        setting = cf.EMP_CNN1d_BL_SETTING[ws]
    else:
        setting = cf.EMP_CNN_BL_SETTING[ws]
    
    filter_size_list, stride_list, dilation_list, max_pooling_list = setting
    
    # 모델 객체 생성
    model_obj = cnn_model.Model(
        ws=ws,
        layer_number=layer_number,
        inplanes=inplanes,
        drop_prob=drop_prob,
        filter_size_list=filter_size_list,
        stride_list=stride_list,
        dilation_list=dilation_list,
        max_pooling_list=max_pooling_list,
        batch_norm=batch_norm,
        xavier=xavier,
        lrelu=lrelu,
        ts1d_model=ts1d_model,
        bn_loc="bn_bf_relu",
        regression_label=regression_label
    )
    
    return model_obj

def process_and_save_results(exp: Experiment) -> None:
    """
    앙상블 결과를 처리하고 저장합니다.

    Args:
        exp (Experiment): 실험 객체
    """
    try:
        # 앙상블 결과 로드 (IS와 OOS 모두 포함)
        ensem_res = load_ensemble_res_with_period_ret(
            year=None,  # 모든 연도 로드
            freq=exp.train_freq,
            country=exp.country,
            ensem_res_dir=exp.ensem_res_dir,
            ensem=exp.ensem,
            ws=exp.ws,
            pw=exp.pw,
            ohlc_len=exp.ohlc_len
        )

        # 상승확률 메트릭 계산 (필요에 따라)
        # up_prob_metrics = calculate_oos_up_prob(ensem_res)

        # 앙상블 결과에서 필요한 컬럼만 선택
        ensem_res = ensem_res.reset_index()  # 멀티인덱스를 컬럼으로 변환

        # Date를 datetime으로 변환
        ensem_res['Date'] = pd.to_datetime(ensem_res['Date'])

        # S&P 500 인덱스 데이터에서 거래일 로드
        trading_days = pd.read_csv(
            os.path.join(dcf.FILTERED_DATA_DIR, 'snp500_index.csv'),
            parse_dates=['Date']
        )['Date'].sort_values().values

        # 다음 거래일 매핑 함수
        def get_next_trading_day(date):
            next_days = trading_days[trading_days > date]
            return next_days[0] if len(next_days) > 0 else None

        # ending_date의 다음 거래일을 investment_date로 설정
        ensem_res['investment_date'] = ensem_res['Date'].apply(get_next_trading_day)

        # None 값이 있는 경우 제거 (마지막 거래일 이후의 데이터)
        ensem_res = ensem_res.dropna(subset=['investment_date'])

        # 최종 컬럼 선택 및 정렬
        ensem_res = ensem_res[['investment_date', 'StockID', 'up_prob']]
        ensem_res = ensem_res.sort_values(['investment_date', 'StockID'])

        # StockID를 문자열로 변환
        ensem_res['StockID'] = ensem_res['StockID'].astype(str)

        # symbol_permno.csv 로드 및 전처리
        symbol_permno = pd.read_csv(os.path.join(dcf.RAW_DATA_DIR, 'symbol_permno.csv'))
        symbol_permno['PERMNO'] = symbol_permno['PERMNO'].astype(str)  # PERMNO를 문자열로 변환
        symbol_permno.rename(columns={'PERMNO': 'StockID'}, inplace=True)  # PERMNO 컬럼을 StockID로 변경

        # 매핑 수행
        ensem_res = ensem_res.merge(symbol_permno, on='StockID', how='left')

        # Symbol이 없는 행 제거
        ensem_res = ensem_res.dropna(subset=['Symbol'])

        # 필요한 컬럼만 선택하고 investment_date를 인덱스로 설정
        ensem_res = ensem_res[['investment_date', 'Symbol', 'up_prob']]
        ensem_res.set_index('investment_date', inplace=True)

        # pivot 테이블 생성
        ensem_res = ensem_res.pivot_table(index=ensem_res.index, columns='Symbol', values='up_prob')

        # 결과 저장
        output_path = os.path.join(exp.model_dir, 'ensem_res.csv')
        ensem_res.to_csv(output_path)
        print(f"Processed ensemble results saved to {output_path}")
        print(f"Final shape: {ensem_res.shape}")

        # 메트릭 저장 (필요에 따라)
        # metrics_dir = Path(exp.model_dir) / "metrics"
        # metrics_dir.mkdir(exist_ok=True)
        # exp_params = {
        #     "model_info": {
        #         "window_size": exp.ws,
        #         "prediction_window": exp.pw,
        #         "train_freq": exp.train_freq,
        #         "ensemble_size": exp.ensem,
        #     },
        #     "metrics": up_prob_metrics.to_dict() if up_prob_metrics is not None else {}
        # }
        # save_exp_params_to_yaml(str(metrics_dir), exp_params)
        # print(f"Results processed and saved to {metrics_dir}")

    except Exception as e:
        print(f"Error processing results: {str(e)}")
        raise

if __name__ == "__main__":
    # GPU ID를 명령줄 인자로 받기
    if len(sys.argv) < 2:
        print("Usage: python experiment.py <GPU_IDS>")
        sys.exit(1)
    
    gpu_ids = sys.argv[1]
    device = set_device(gpu_ids)
    print(f"Using device: {device}")

    # CNN2D 모델 학습
    print("\nTraining CNN2D model...")
    ws = 20
    pw = 20
    
    # 모델 객체 생성
    model_obj = create_model_object(
        ws=ws,
        ts1d_model=False,  # CNN2D
        drop_prob=0.50,
        batch_norm=True,
        xavier=True,
        lrelu=True
    )
    
    # 실험 객체 생성
    exp = Experiment(
        ws=ws,
        pw=pw,
        model_obj=model_obj,
        train_freq="month",
        ensem=5,
        lr=1e-5,
        drop_prob=0.50,
        max_epoch=500,
        enable_tqdm=True,
        early_stop=True,
        has_ma=True,
        has_volume_bar=True,
        is_years=cf.IS_YEARS,
        oos_years=cf.OOS_YEARS,
        country="USA",
        chart_type="bar",
        delayed_ret=0
    )

    # 모델 학습
    exp.train_empirical_ensem_model()

    # 포트폴리오 계산
    exp.calculate_portfolio(
        load_saved_data=True,
        delay_list=[0],
        is_ensem_res=True,
        cut=10
    )
    
    # 결과 처리 및 저장
    print("Processing and saving results...")
    process_and_save_results(exp)
    
    print("실험이 성공적으로 완료되었습니다.")