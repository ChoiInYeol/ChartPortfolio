"""데이터 로딩을 위한 모듈입니다."""

import pandas as pd
import numpy as np
import logging
import os
from typing import Dict
import glob

class DataLoader:
    """데이터 로딩을 위한 클래스입니다."""
    
    def __init__(self, base_folder: str, data_size: int, train_date: str = '2017-12-31', end_date: str = '2024-07-05', ws: int = 20, pw: int = 20):
        """
        DataLoader 초기화
        
        Args:
            base_folder (str): 기본 폴더 경로
            data_size (int): 데이터 크기 (50, 370, 500, 2055)
            train_date (str): 학습 시작 날짜
            end_date (str): 투자 종료 날짜
            ws (int): 학습 윈도우 크기
            pw (int): 예측 윈도우 크기
        """
        self.base_folder = base_folder
        self.data_size = data_size
        self.ws = ws    
        self.pw = pw
        self.train_date = train_date
        self.end_date = end_date
        self.logger = logging.getLogger(__name__)
        
        # 결과 디렉토리 설정
        self.result_dir = os.path.join(
            base_folder,
            'Backtest',
            'results',
            f'size_{data_size}'
        )
        
        # 하위 디렉토리 생성
        for subdir in ['weights', 'figures', 'metrics']:
            os.makedirs(os.path.join(self.result_dir, subdir), exist_ok=True)

    def load_returns(self) -> pd.DataFrame:
        """수익률 데이터를 로드합니다."""
        try:
            # 파일 패턴 확인
            returns_pattern = f'filtered_returns_{self.data_size}_*{self.ws}D{self.pw}P.csv'
            
            # 디렉토리 내 파일 검색
            ts_model_dir = os.path.join(self.base_folder, 'TS_Model', 'data')
            returns_files = glob.glob(os.path.join(ts_model_dir, returns_pattern))
            
            if not returns_files:
                raise FileNotFoundError(
                    f"Returns data file not found for ws={self.ws}, pw={self.pw}"
                )
            
            # 가장 최신 파일 사용
            returns_path = max(returns_files, key=os.path.getctime)
            
            # 데이터 로드
            returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
            
            # 날짜 필터링 및 결측치 처리
            returns = returns[(returns.index >= self.train_date) & 
                            (returns.index <= self.end_date)]
            returns = returns.ffill().bfill()
            
            return returns
            
        except Exception as e:
            self.logger.error(f"Error loading returns data: {str(e)}")
            raise

    def load_probabilities(self) -> pd.DataFrame:
        """상승확률 데이터를 로드합니다."""
        try:
            # 파일 패턴 확인
            probs_pattern = f'filtered_probs_{self.data_size}_*{self.ws}D{self.pw}P.csv'
            
            # 디렉토리 내 파일 검색
            ts_model_dir = os.path.join(self.base_folder, 'TS_Model', 'data')
            probs_files = glob.glob(os.path.join(ts_model_dir, probs_pattern))
            
            if not probs_files:
                raise FileNotFoundError(
                    f"Probability data file not found for ws={self.ws}, pw={self.pw}"
                )
            
            # 가장 최신 파일 사용
            probs_path = max(probs_files, key=os.path.getctime)
            
            # 데이터 로드
            probs = pd.read_csv(probs_path, index_col=0, parse_dates=True)
            
            # 날짜 필터링 및 결측치 처리
            probs = probs[(probs.index >= self.train_date) & 
                         (probs.index <= self.end_date)]
            probs = probs.fillna(0.5)
            
            return probs
            
        except Exception as e:
            self.logger.error(f"Error loading probability data: {str(e)}")
            raise

    def get_result_path(self, subdir: str, filename: str) -> str:
        """결과 파일의 전체 경로를 반환합니다."""
        return os.path.join(self.result_dir, subdir, filename)

    def load_benchmark(self) -> pd.Series:
        """S&P 500 지수 데이터를 로드합니다."""
        try:
            benchmark_path = os.path.join(self.base_folder, 'Data', 'processed', 'snp500_index.csv')
            
            self.logger.info(f"Loading benchmark data from: {benchmark_path}")
            
            benchmark = pd.read_csv(benchmark_path, index_col=0, parse_dates=True)
            benchmark_returns = benchmark['Adj Close'].pct_change()
            
            # 날짜 필터링
            benchmark_returns = benchmark_returns[
                (benchmark_returns.index >= self.train_date) & 
                (benchmark_returns.index <= self.end_date)
            ]
            
            return benchmark_returns
            
        except Exception as e:
            self.logger.error(f"Error loading benchmark data: {str(e)}")
            raise 