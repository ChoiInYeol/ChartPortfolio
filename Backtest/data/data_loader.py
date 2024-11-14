"""
데이터 로딩을 위한 모듈입니다.
주식 수익률, 벤치마크, 앙상블 결과 등을 로드하는 기능을 포함합니다.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional
import glob

class DataLoader:
    """
    데이터 로딩을 위한 클래스입니다.
    """
    
    def __init__(self, base_folder: str, train_date: str = '2017-12-31'):
        """
        DataLoader 초기화
        
        Args:
            base_folder (str): 기본 데이터 폴더 경로
            train_date (str): 학습 시작 날짜
        """
        self.base_folder = base_folder
        self.train_date = train_date
        self.logger = logging.getLogger(__name__)
        
        # 데이터 경로 설정
        self.data_path = os.path.join(self.base_folder, 'data')

    def load_stock_returns(self) -> pd.DataFrame:
        """주식 수익률 데이터를 로드합니다."""
        file_path = os.path.join(self.base_folder, '..', 'POD', 'data', 'return_df.csv')
        if not os.path.exists(file_path):
            self.logger.error(f"Return data file not found: {file_path}")
            raise FileNotFoundError(f"Return data file not found: {file_path}")
            
        returns = pd.read_csv(file_path, index_col=0)
        returns.index = pd.to_datetime(returns.index)
        returns = returns[returns.index >= self.train_date]
        
        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns = returns.ffill().fillna(0)
        
        # 인덱스 정렬
        returns = returns.sort_index()
        
        return returns

    def load_up_prob(self, model: str = 'CNN', freq: str = 'month', 
                     ws: int = 20, pw: int = 20) -> pd.DataFrame:
        """
        상승확률 데이터를 로드합니다.
        
        Args:
            model (str): 모델 이름 (예: CNN, GRU)
            freq (str): 주기 (예: month, week)
            ws (int): 윈도우 사이즈
            pw (int): 예측 윈도우
        """
        filename = f"{model}_{freq}_{ws}D_{pw}P_up_prob_df.csv"
        file_path = os.path.join(self.data_path, filename)
        
        if not os.path.exists(file_path):
            self.logger.error(f"Up probability data file not found: {file_path}")
            raise FileNotFoundError(f"Up probability data file not found: {file_path}")
            
        up_prob = pd.read_csv(file_path, index_col=0)
        up_prob.index = pd.to_datetime(up_prob.index)
        up_prob = up_prob[up_prob.index >= self.train_date]
        
        return up_prob

    def load_benchmark(self) -> pd.DataFrame:
        """벤치마크 데이터를 로드합니다."""
        file_path = os.path.join(self.data_path, 'snp500_index.csv')
        if not os.path.exists(file_path):
            self.logger.error(f"Benchmark data file not found: {file_path}")
            raise FileNotFoundError(f"Benchmark data file not found: {file_path}")
            
        snp500 = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        snp500.sort_index(inplace=True)
        snp500 = snp500[snp500.index >= self.train_date]
        snp500['Returns'] = snp500['Adj Close'].pct_change()
        snp500['Cumulative Returns'] = (1 + snp500['Returns']).cumprod()
        
        return snp500

    def load_model_weights(self, model: str, use_prob: bool) -> Dict[str, pd.DataFrame]:
        """모델 가중치를 로드합니다."""
        weights_pattern = os.path.join(
            self.base_folder, '..', 'POD', 'Result', 
            model, f'weights_{str(use_prob)}_{model}_*.csv'
        )
        weights_files = glob.glob(weights_pattern)
        
        if not weights_files:
            self.logger.warning(f"No weights found for {model} (use_prob={use_prob})")
            return {}
            
        weights_dict = {}
        for file in weights_files:
            try:
                weights = pd.read_csv(file, index_col=0)
                weights.index = pd.to_datetime(weights.index)
                weights = weights[weights.index >= self.train_date]
                
                filename = os.path.basename(file)
                parts = filename.split('_')
                n_select = int(parts[3][1:])
                
                key = f"{model} Top {n_select}"
                    
                weights_dict[key] = weights
                
                self.logger.info(f"Loaded weights from {filename}")
                self.logger.info(f"Shape: {weights.shape}, Date range: {weights.index[0]} to {weights.index[-1]}")
                
            except Exception as e:
                self.logger.error(f"Error loading weights from {file}: {str(e)}")
                continue
                
        return weights_dict