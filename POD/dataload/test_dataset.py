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
    with open("data/dataloaders.pkl", "rb") as f:
        data_dict = pickle.load(f)
    
    # 필수 키 확인
    required_keys = ['train_x', 'train_y', 'train_prob', 'train_dates',
                    'val_x', 'val_y', 'val_prob', 'val_dates',
                    'test_x', 'test_y', 'test_prob', 'test_dates']
    
    for key in required_keys:
        assert key in data_dict, f"Missing key: {key}"
        
    # 데이터 형태 검증
    for prefix in ['train', 'val', 'test']:
        x = data_dict[f'{prefix}_x']
        y = data_dict[f'{prefix}_y']
        prob = data_dict[f'{prefix}_prob']
        dates = data_dict[f'{prefix}_dates']
        
        # 차원 검증
        assert x.ndim == 3, f"{prefix}_x should be 3D: (batch, seq_len, n_stocks)"
        assert y.ndim == 3, f"{prefix}_y should be 3D: (batch, pred_len, n_stocks)"
        assert prob.ndim == 2, f"{prefix}_prob should be 2D: (batch, n_stocks)"
        
        # 배치 크기 일관성 검증
        batch_size = len(x)
        assert len(y) == batch_size, f"Inconsistent batch size in {prefix}_y"
        assert len(prob) == batch_size, f"Inconsistent batch size in {prefix}_prob"
        assert len(dates) == batch_size, f"Inconsistent batch size in {prefix}_dates"
        
        # 값 범위 검증
        assert not np.isnan(x).any(), f"NaN values found in {prefix}_x"
        assert not np.isnan(y).any(), f"NaN values found in {prefix}_y"
        assert not np.isnan(prob).any(), f"NaN values found in {prefix}_prob"
        
        logger.info(f"{prefix} dataset shapes:")
        logger.info(f"x: {x.shape}")
        logger.info(f"y: {y.shape}")
        logger.info(f"prob: {prob.shape}")
        logger.info(f"dates length: {len(dates)}")

def test_dataset_values():
    """데이터셋 값의 유효성 검증"""
    with open("data/dataloaders.pkl", "rb") as f:
        data_dict = pickle.load(f)
    
    for prefix in ['train', 'val', 'test']:
        x = data_dict[f'{prefix}_x']
        y = data_dict[f'{prefix}_y']
        prob = data_dict[f'{prefix}_prob']
        
        # 확률 값 범위 검증
        assert (prob >= 0).all() and (prob <= 1).all(), f"Invalid probability values in {prefix}_prob"
        
        # 수익률 값 범위 검증 (극단값 체크)
        assert abs(x).max() < 1.0, f"Extreme return values found in {prefix}_x"
        assert abs(y).max() < 1.0, f"Extreme return values found in {prefix}_y"
        
        # 시계열 연속성 검증
        for i in range(len(x)):
            assert np.allclose(x[i, -1], y[i, 0], atol=1e-5), \
                f"Time series discontinuity at index {i} in {prefix} dataset"

def test_date_consistency():
    """날짜 데이터의 일관성 검증"""
    with open("data/dataloaders.pkl", "rb") as f:
        data_dict = pickle.load(f)
    
    for prefix in ['train', 'val', 'test']:
        dates = pd.to_datetime(data_dict[f'{prefix}_dates'])
        
        # 날짜 순서 검증
        assert dates.is_monotonic_increasing, f"Dates are not in order in {prefix}_dates"
        
        # 날짜 간격 검증
        date_diffs = dates.diff()[1:]
        assert (date_diffs == pd.Timedelta(days=1)).all(), \
            f"Non-consecutive dates found in {prefix}_dates"
        
        logger.info(f"{prefix} date range: {dates.min()} to {dates.max()}")

def test_return_distribution():
    """수익률 분포 분석"""
    with open("data/dataloaders.pkl", "rb") as f:
        data_dict = pickle.load(f)
    
    for prefix in ['train', 'val', 'test']:
        x = data_dict[f'{prefix}_x']
        y = data_dict[f'{prefix}_y']
        
        # 수익률 기본 통계량
        logger.info(f"\n{prefix} return statistics:")
        logger.info(f"X - Mean: {x.mean():.4f}, Std: {x.std():.4f}")
        logger.info(f"X - Min: {x.min():.4f}, Max: {x.max():.4f}")
        logger.info(f"X - 1%: {np.percentile(x, 1):.4f}, 99%: {np.percentile(x, 99):.4f}")
        
        logger.info(f"Y - Mean: {y.mean():.4f}, Std: {y.std():.4f}")
        logger.info(f"Y - Min: {y.min():.4f}, Max: {y.max():.4f}")
        logger.info(f"Y - 1%: {np.percentile(y, 1):.4f}, 99%: {np.percentile(y, 99):.4f}")
        
        # 극단값 비율 확인
        extreme_threshold = 0.5  # 50% 이상의 수익률을 극단값으로 간주
        x_extreme_ratio = (abs(x) > extreme_threshold).mean()
        y_extreme_ratio = (abs(y) > extreme_threshold).mean()
        
        logger.info(f"Extreme values ratio (|return| > {extreme_threshold}):")
        logger.info(f"X: {x_extreme_ratio:.4%}")
        logger.info(f"Y: {y_extreme_ratio:.4%}")
        
        # 기본적인 검증
        assert not np.isnan(x).any(), f"NaN values found in {prefix}_x"
        assert not np.isnan(y).any(), f"NaN values found in {prefix}_y"
        assert not np.isinf(x).any(), f"Inf values found in {prefix}_x"
        assert not np.isinf(y).any(), f"Inf values found in {prefix}_y"
        
        # 수익률의 연속성 검증 (시계열 특성)
        for i in range(len(x)):
            assert np.allclose(x[i, -1], y[i, 0], atol=1e-5), \
                f"Time series discontinuity at index {i} in {prefix} dataset"

if __name__ == "__main__":
    logger.info("Starting dataset validation...")
    
    try:
        test_dataset_structure()
        logger.info("Dataset structure validation passed")
        
        test_return_distribution()
        logger.info("Return distribution analysis completed")
        
        test_date_consistency()
        logger.info("Date consistency validation passed")
        
    except AssertionError as e:
        logger.error(f"Validation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}") 