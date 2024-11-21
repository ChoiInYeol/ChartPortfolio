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

def process_ensemble_results(exp: Experiment):
    """
    앙상블 결과를 처리하고 ImagePortOpt/Data 폴더에 저장합니다.
    
    Args:
        exp (Experiment): 실험 객체
    """
    import pandas as pd
    import logging
    from datetime import datetime
    from pathlib import Path
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 현재 실행 경로(CNN_Model)에서 상위로 올라가서 Data 폴더 경로 설정
    current_path = Path(__file__).resolve()  # experiment.py의 절대 경로
    project_root = current_path.parent.parent  # ImagePortOpt 폴더
    data_dir = project_root / 'Data'
    data_dir.mkdir(exist_ok=True)
    
    logger.info(f"Data directory: {data_dir}")
    
    # 앙상블 결과 폴더 경로
    ensem_res_dir = Path(exp.ensem_res_dir)
    logger.info(f"Ensemble results directory: {ensem_res_dir}")
    
    try:
        # 해당 폴더의 모든 CSV 파일 처리
        csv_files = list(ensem_res_dir.glob('*.csv'))
        if not csv_files:
            logger.error(f"No CSV files found in {ensem_res_dir}")
            return
            
        all_results = []
        for csv_file in csv_files:
            logger.info(f'Processing {csv_file.name}...')
            
            df = pd.read_csv(
                csv_file,
                parse_dates=['ending_date']
            )
            
            # 필요한 컬럼만 선택
            df = df[['ending_date', 'StockID', 'up_prob', 'ret_val']]
            
            # 컬럼명 변경
            df = df.rename(columns={
                'ending_date': 'investment_date',
                'up_prob': f'up_prob_CNN{exp.ws}'
            })
            
            all_results.append(df)
            
        # 모든 결과 병합
        final_df = pd.concat(all_results, ignore_index=True)
        
        # 중복 제거 (같은 날짜, 같은 종목에 대해)
        final_df = final_df.drop_duplicates(subset=['investment_date', 'StockID'])
        
        # 정렬
        final_df = final_df.sort_values(['investment_date', 'StockID'])
        
        # 저장
        output_filename = f'ensemble_results_CNN{exp.ws}_{datetime.now().strftime("%Y%m%d")}.parquet'
        output_path = data_dir / output_filename
        
        # parquet 형식으로 저장
        final_df.to_parquet(output_path, index=False)
        logger.info(f'Saved combined results to {output_path}')
        logger.info(f'Final shape: {final_df.shape}')
        
        # CSV 형식으로도 저장
        csv_path = data_dir / f'ensemble_results_CNN{exp.ws}_{datetime.now().strftime("%Y%m%d")}.csv'
        final_df.to_csv(csv_path, index=False)
        logger.info(f'Also saved as CSV to {csv_path}')
        
    except Exception as e:
        logger.error(f'Error processing ensemble results: {str(e)}')

def create_up_prob_pivot(exp: Experiment):
    """
    앙상블 결과에서 up_prob를 피벗하여 저장합니다.
    SP500(2018년 기준) 종목만 필터링하여 저장합니다.
    
    Args:
        exp (Experiment): 실험 객체
    """
    import pandas as pd
    from pathlib import Path
    import logging
    from datetime import datetime
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # 현재 실행 경로에서 Data 폴더 경로 설정
        current_path = Path(__file__).resolve()
        project_root = current_path.parent.parent
        data_dir = project_root / 'Data'
        
        # SP500 종목 로드
        sp500_df = pd.read_csv(data_dir / 'sp500_20180101.csv')
        sp500_symbols = set(sp500_df['Symbol'].tolist())
        logger.info(f"Loaded {len(sp500_symbols)} SP500 symbols")
        
        # 최신 앙상블 결과 파일 찾기
        ensemble_files = list(data_dir.glob(f'ensemble_results_CNN{exp.ws}_*.csv'))
        if not ensemble_files:
            logger.error("앙상블 결과 파일을 찾을 수 없습니다.")
            return
            
        latest_file = max(ensemble_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Processing {latest_file.name}")
        
        # 데이터 로드
        df = pd.read_csv(latest_file, parse_dates=['investment_date'])
        
        # Symbol-PERMNO 매핑 로드
        symbol_permno = pd.read_csv(data_dir / 'symbol_permno.csv')
        permno_to_symbol = dict(zip(symbol_permno['PERMNO'], symbol_permno['Symbol']))
        
        # PERMNO를 Symbol로 변환
        df['Symbol'] = df['StockID'].map(permno_to_symbol)
        
        # SP500 종목만 필터링
        df = df[df['Symbol'].isin(sp500_symbols)]
        logger.info(f"Filtered to {df['Symbol'].nunique()} SP500 stocks")
        
        # up_prob 피벗
        up_prob_df = df.pivot(
            index='investment_date',
            columns='Symbol',
            values=f'up_prob_CNN{exp.ws}'
        )
        
        # SP500에 있지만 데이터에 없는 종목 확인
        missing_symbols = sp500_symbols - set(up_prob_df.columns)
        if missing_symbols:
            logger.warning(f"Missing {len(missing_symbols)} SP500 symbols: {sorted(missing_symbols)}")
        
        # 파일명 생성
        freq = exp.train_freq if hasattr(exp, 'train_freq') else 'unknown'
        model_dim = "1D" if exp.model_obj.ts1d_model else "2D"
        output_filename = f"{model_dim}_{freq}_{exp.ws}D_{exp.pw}P_up_prob_df.csv"
        output_path = data_dir / output_filename
        
        # 결과 저장
        up_prob_df.to_csv(output_path)
        logger.info(f"Saved up_prob pivot to {output_path}")
        logger.info(f"Model type: {model_dim}")
        logger.info(f"Shape: {up_prob_df.shape}")
        logger.info(f"Period: {up_prob_df.index.min()} ~ {up_prob_df.index.max()}")
        logger.info(f"Number of stocks: {len(up_prob_df.columns)}")
        
    except Exception as e:
        logger.error(f"Error creating up_prob pivot: {str(e)}")

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
        train_freq="week",
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
    
    # 앙상블 결과 처리 및 저장
    process_ensemble_results(exp)

    # 앙상블 결과에서 up_prob를 피벗하여 저장
    create_up_prob_pivot(exp)
