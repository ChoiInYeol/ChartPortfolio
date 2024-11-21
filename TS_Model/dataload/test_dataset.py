import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dataset_structure():
    """데이터셋의 기본 구조 검증"""
    current_path = Path(__file__).resolve()
    data_dir = current_path.parent.parent / 'data'
    
    # 모든 데이터셋 파일 검사
    dataset_files = list(data_dir.glob("dataset_*.pkl"))
    
    for dataset_file in dataset_files:
        logger.info(f"\nTesting dataset: {dataset_file.name}")
        
        # 파일명에서 정보 추출
        parts = dataset_file.stem.split('_')
        model_type = parts[1]  # 1D or 2D
        period = parts[2]      # day, week, month, quarter
        ws = 60 # int(parts[3].replace('D', ''))
        pw = int(parts[4].replace('P', ''))
        
        with open(dataset_file, "rb") as f:
            data_dict = pickle.load(f)
        
        # 필수 키 확인
        required_keys = ['train', 'val', 'test']
        for key in required_keys:
            assert key in data_dict, f"Missing key: {key}"
            assert len(data_dict[key]) == 3, f"Invalid tuple length for {key}"
            x, y, prob = data_dict[key]
            
            # 차원 검증
            assert x.ndim == 3, f"{key}_x should be 3D: (batch, seq_len, n_stocks)"
            assert y.ndim == 3, f"{key}_y should be 3D: (batch, pred_len, n_stocks)"
            assert prob.ndim == 3, f"{key}_prob should be 3D: (batch, pred_len, n_stocks)"
            
            # 배치 크기 일관성 검증
            batch_size = len(x)
            assert len(y) == batch_size, f"Inconsistent batch size in {key}_y"
            assert len(prob) == batch_size, f"Inconsistent batch size in {key}_prob"
            
            # 시퀀스 길이 검증
            assert x.shape[1] == ws, f"Invalid sequence length in {key}_x: {x.shape[1]} != {ws}"
            assert y.shape[1] == pw, f"Invalid prediction length in {key}_y: {y.shape[1]} != {pw}"
            assert prob.shape[1] == pw, f"Invalid prediction length in {key}_prob: {prob.shape[1]} != {pw}"
            
            logger.info(f"{key} dataset shapes:")
            logger.info(f"x: {x.shape}")
            logger.info(f"y: {y.shape}")
            logger.info(f"prob: {prob.shape}")

def test_dataset_values():
    """데이터셋 값의 유효성 검증"""
    current_path = Path(__file__).resolve()
    data_dir = current_path.parent.parent / 'data'
    
    for dataset_file in data_dir.glob("dataset_*.pkl"):
        logger.info(f"\nTesting values in: {dataset_file.name}")
        
        with open(dataset_file, "rb") as f:
            data_dict = pickle.load(f)
        
        for key in ['train', 'val', 'test']:
            x, y, prob = data_dict[key]
            
            # 수익률 값 범위 검증
            assert not np.isnan(x).any(), f"NaN values found in {key}_x"
            assert not np.isnan(y).any(), f"NaN values found in {key}_y"
            assert not np.isnan(prob).any(), f"NaN values found in {key}_prob"
            assert not np.isinf(x).any(), f"Inf values found in {key}_x"
            assert not np.isinf(y).any(), f"Inf values found in {key}_y"
            assert not np.isinf(prob).any(), f"Inf values found in {key}_prob"
            
            # 확률 값 범위 검증
            assert (prob >= 0).all() and (prob <= 1).all(), \
                f"Invalid probability values in {key}_prob: min={prob.min()}, max={prob.max()}"
            
            # 수익률 값 범위 검증 (극단값 체크)
            assert abs(x).max() < 1.0, \
                f"Extreme return values found in {key}_x: max={abs(x).max()}"
            assert abs(y).max() < 1.0, \
                f"Extreme return values found in {key}_y: max={abs(y).max()}"
            
            # 기초 통계량 출력
            logger.info(f"\n{key} statistics:")
            logger.info(f"Returns X - Mean: {x.mean():.4f}, Std: {x.std():.4f}")
            logger.info(f"Returns X - Min: {x.min():.4f}, Max: {x.max():.4f}")
            logger.info(f"Returns Y - Mean: {y.mean():.4f}, Std: {y.std():.4f}")
            logger.info(f"Returns Y - Min: {y.min():.4f}, Max: {y.max():.4f}")
            logger.info(f"Prob - Mean: {prob.mean():.4f}, Std: {prob.std():.4f}")
            logger.info(f"Prob - Min: {prob.min():.4f}, Max: {prob.max():.4f}")

def test_dates_consistency():
    """날짜 데이터의 일관성 검증"""
    current_path = Path(__file__).resolve()
    data_dir = current_path.parent.parent / 'data'
    
    for dataset_file in data_dir.glob("dates_*.pkl"):
        logger.info(f"\nTesting dates in: {dataset_file.name}")
        
        with open(dataset_file, "rb") as f:
            dates_dict = pickle.load(f)
        
        for key in ['train', 'val', 'test']:
            dates = dates_dict[key]
            
            # 날짜 형식 검증
            assert all(isinstance(date_list, list) for date_list in dates), \
                f"Invalid date format in {key}"
            
            # 날짜 순서 검증
            for date_list in dates:
                date_series = pd.to_datetime(date_list)
                assert date_series.is_monotonic_increasing, \
                    f"Dates are not in order in {key}"
                
                # 날짜 간격 검증
                date_diffs = date_series.diff()[1:]
                assert (date_diffs == pd.Timedelta(days=1)).all(), \
                    f"Non-consecutive dates found in {key}"
            
            logger.info(f"{key} dates range: {dates[0][0]} to {dates[-1][-1]}")
            logger.info(f"Number of sequences: {len(dates)}")

if __name__ == "__main__":
    logger.info("Starting dataset validation...")
    
    try:
        test_dataset_structure()
        logger.info("Dataset structure validation passed")
        
        test_dataset_values()
        logger.info("Dataset values validation passed")
        
        test_dates_consistency()
        logger.info("Dates consistency validation passed")
        
    except AssertionError as e:
        logger.error(f"Validation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}") 