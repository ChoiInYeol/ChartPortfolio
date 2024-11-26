import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_random_benchmark(n_select: int = 100, seed: int = 42):
    """
    return_df와 up_prob_df에서 랜덤하게 n_select개의 종목을 선택하여 새로운 데이터셋 생성
    
    Args:
        n_select (int): 선택할 종목 수 (기본값: 100)
        seed (int): 랜덤 시드 (기본값: 42)
    """
    try:
        # 데이터 경로 설정
        data_dir = Path(__file__).parent
        returns_path = data_dir / 'return_df.csv'
        probs_path = data_dir / '2D_month_20D_20P_up_prob_df.csv'
        
        # 데이터 로드
        returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        probs_df = pd.read_csv(probs_path, index_col=0, parse_dates=True)
        
        logger.info(f"원본 데이터 shape - returns: {returns_df.shape}, probs: {probs_df.shape}")
        
        # 공통 종목만 선택
        common_symbols = sorted(list(set(returns_df.columns) & set(probs_df.columns)))
        returns_df = returns_df[common_symbols]
        probs_df = probs_df[common_symbols]
        
        logger.info(f"공통 종목 수: {len(common_symbols)}")
        
        # 랜덤 종목 선택
        np.random.seed(seed)
        selected_symbols = np.random.choice(common_symbols, size=n_select, replace=False)
        selected_symbols = sorted(selected_symbols)  # 정렬하여 일관성 유지
        
        # 선택된 종목으로 데이터프레임 생성
        selected_returns = returns_df[selected_symbols]
        selected_probs = probs_df[selected_symbols]
        
        # 결과 저장
        output_returns = data_dir / f'random_return_df_{n_select}.csv'
        output_probs = data_dir / f'random_up_prob_df_{n_select}.csv'
        
        selected_returns.to_csv(output_returns)
        selected_probs.to_csv(output_probs)
        
        logger.info(f"선택된 {n_select}개 종목 데이터 저장 완료")
        logger.info(f"- Returns 저장 경로: {output_returns}")
        logger.info(f"- Probs 저장 경로: {output_probs}")
        
        return selected_returns, selected_probs
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    create_random_benchmark(n_select=100, seed=42)
