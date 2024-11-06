"""
데이터 로딩을 위한 모듈입니다.
주식 수익률, 벤치마크, 앙상블 결과 등을 로드하는 기능을 포함합니다.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional
from Data import dgp_config as dcf

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
        self.work_folder = os.path.join(base_folder, 'WORK_DIR')
        self.train_date = train_date
        self.logger = logging.getLogger(__name__)

    def load_stock_returns(self) -> pd.DataFrame:
        """
        주식 수익률 데이터를 로드합니다.
        
        Returns:
            pd.DataFrame: 전처리된 수익률 데이터
        """
        file_path = os.path.join(self.base_folder, 'processed_data', 'us_ret.feather')
        us_ret = pd.read_feather(file_path)
        us_ret = us_ret[us_ret['Date'] >= self.train_date]
        us_ret = us_ret.pivot(index='Date', columns='StockID', values='Ret')
        us_ret.index = pd.to_datetime(us_ret.index)

        if us_ret.abs().mean().mean() > 1:
            self.logger.info("Converting returns from percentages to decimals.")
            us_ret = us_ret / 100

        # 결측치 처리
        us_ret = us_ret.replace([np.inf, -np.inf], np.nan)
        us_ret = us_ret.fillna(method='ffill').fillna(0)

        return us_ret

    def load_benchmark(self) -> pd.DataFrame:
        """
        벤치마크 데이터를 로드합니다.
        
        Returns:
            pd.DataFrame: S&P 500 벤치마크 데이터
        """
        snp500 = pd.read_csv(os.path.join(dcf.RAW_DATA_DIR, "snp500_index.csv"), parse_dates=['Date'], index_col='Date')
        snp500.sort_index(inplace=True)
        snp500 = snp500[snp500.index >= self.train_date]
        snp500['Returns'] = snp500['Adj Close'].pct_change()
        snp500['Cumulative Returns'] = (1 + snp500['Returns']).cumprod()
        
        return snp500

    def load_ensemble_results(self, model: str, window_size: int) -> pd.DataFrame:
        """
        앙상블 결과를 로드합니다. 파일이 없으면 처리 후 저장합니다.
        
        Args:
            model (str): 모델 이름
            window_size (int): 윈도우 크기
            
        Returns:
            pd.DataFrame: 앙상블 결과 데이터
        """
        file_path = os.path.join(
            self.base_folder,
            'WORK_DIR',
            f'ensemble_{model}{window_size}_res.csv'
        )
        
        if not os.path.exists(file_path):
            self.logger.warning(f"Ensemble results file not found: {file_path}")
            self.logger.info("Processing ensemble results...")
            
            # DataProcessor 인스턴스 생성 및 앙상블 결과 처리
            from .data_processor import DataProcessor
            processor = DataProcessor()
            ensemble_results = processor.process_ensemble_results(
                self.base_folder, model, window_size
            )
            
            if ensemble_results.empty:
                self.logger.error("Failed to process ensemble results")
                return pd.DataFrame()
                
            return ensemble_results

        ensemble_results = pd.read_csv(file_path)
        
        if isinstance(ensemble_results.index, pd.MultiIndex):
            ensemble_results = ensemble_results.reset_index()
        
        ensemble_results['investment_date'] = pd.to_datetime(
            ensemble_results['investment_date']
        )
        ensemble_results['ending_date'] = pd.to_datetime(
            ensemble_results['ending_date']
        )
        ensemble_results['StockID'] = ensemble_results['StockID'].astype(str)
        
        return ensemble_results