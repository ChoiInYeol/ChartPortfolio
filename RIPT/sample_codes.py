from Experiments.cnn_experiment import Experiment
from Model import cnn_model
from Misc import config as cf

from typing import Optional
import torch
import os
import sys

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

if __name__ == "__main__":
    # GPU ID를 명령줄 인자로 받기
    if len(sys.argv) < 2:
        print("Usage: python sample_codes.py <GPU_IDS>")
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
        chart_type="bar"
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
