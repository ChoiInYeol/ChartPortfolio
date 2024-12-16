"""데이터 로딩을 위한 모듈입니다."""

import pandas as pd
import numpy as np
import logging
import os
import yaml
from typing import Dict
import glob

class DataLoader:
    """데이터 로딩을 위한 클래스입니다."""
    
    def __init__(self, base_folder: str, data_size: int, train_date: str = '2017-12-31', end_date: str = '2024-07-05', ws: int = 60, pw: int = 60):
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
        
        # 설정 파일 로드
        with open(os.path.join(base_folder, 'weight.yaml'), 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 결과 디렉토리 설정
        self.result_dir = os.path.join(
            base_folder,  # base_folder 직접 사용
            'results',
            f'Result_{data_size}_{ws}D{pw}P'
        )
        
        # 하위 디렉토리 생성
        for subdir in ['weights', 'figures']:  # 필요한 하위 디렉토리만 명시적으로 지정
            os.makedirs(os.path.join(self.result_dir, subdir), exist_ok=True)

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """필요한 데이터를 모두 로드합니다."""
        try:
            # 파일 패턴 확인
            returns_pattern = f'filtered_returns_{self.data_size}_*{self.ws}D{self.pw}P.csv'
            probs_pattern = f'filtered_probs_{self.data_size}_*{self.ws}D{self.pw}P.csv'
            
            # 디렉토리 내 파일 검색
            ts_model_dir = self.config['paths']['ts_model_data']
            returns_files = glob.glob(os.path.join(ts_model_dir, returns_pattern))
            probs_files = glob.glob(os.path.join(ts_model_dir, probs_pattern))
            
            if not returns_files or not probs_files:
                raise FileNotFoundError(
                    f"Required data files not found for ws={self.ws}, pw={self.pw}"
                )
            
            # 가장 최신 파일 사용
            returns_path = max(returns_files, key=os.path.getctime)
            probs_path = max(probs_files, key=os.path.getctime)
            
            # 데이터 로드
            returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
            probs = pd.read_csv(probs_path, index_col=0, parse_dates=True)
            
            # 날짜 필터링 및 결측치 처리
            returns = returns[(returns.index >= self.train_date) & 
                            (returns.index <= self.end_date)]
            returns = returns.ffill().bfill()
            
            probs = probs[(probs.index >= self.train_date) & 
                         (probs.index <= self.end_date)]
            probs = probs.fillna(0.5)
            
            return {
                'returns': returns,
                'probs': probs
            }
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def load_ts_model_weights(self, model_name: str) -> Dict[str, pd.DataFrame]:
        """시계열 모델의 포트폴리오 가중치를 로드합니다."""
        weights_dict = {}
        
        try:
            # 절대 경로 사용
            project_root = "/home/indi/codespace/ImagePortOpt"
            ts_model_dir = os.path.join(project_root, 'TS_Model')
            
            self.logger.info(f"Current working directory: {os.getcwd()}")
            self.logger.info(f"Base folder: {self.base_folder}")
            self.logger.info(f"Project root: {project_root}")
            self.logger.info(f"TS Model dir: {ts_model_dir}")
            
            # 가중치 파일 전체 경로 (noprob)
            noprob_pattern = os.path.join(
                ts_model_dir,
                'Result',
                f'{model_name}_{self.data_size}_{self.ws}D{self.pw}P',
                f'portfolio_weights_{model_name}_top{self.data_size}_noprob_maxsharpe.csv'
            )
            
            # 가중치 파일 전체 경로 (prob)
            prob_pattern = os.path.join(
                ts_model_dir,
                'Result',
                f'{model_name}_{self.data_size}_{self.ws}D{self.pw}P',
                f'portfolio_weights_{model_name}_top{self.data_size}_prob_maxsharpe.csv'
            )
            
            self.logger.info(f"Checking file existence:")
            self.logger.info(f"noprob exists: {os.path.exists(noprob_pattern)}")
            self.logger.info(f"prob exists: {os.path.exists(prob_pattern)}")
            
            # noprob 가중치 로드
            if os.path.exists(noprob_pattern):
                self.logger.info(f"Loading noprob weights from: {noprob_pattern}")
                weights = pd.read_csv(noprob_pattern, index_col=0, parse_dates=True)
                weights = weights[(weights.index >= self.train_date) & 
                                (weights.index <= self.end_date)].fillna(0)
                weights_dict[model_name] = weights
                self.logger.info(f"Successfully loaded {model_name} weights")
            
            # prob 가중치 로드
            if os.path.exists(prob_pattern):
                self.logger.info(f"Loading prob weights from: {prob_pattern}")
                weights = pd.read_csv(prob_pattern, index_col=0, parse_dates=True)
                weights = weights[(weights.index >= self.train_date) & 
                                (weights.index <= self.end_date)].fillna(0)
                weights_dict[f'CNN{model_name}'] = weights
                self.logger.info(f"Successfully loaded CNN{model_name} weights")
            
            if not weights_dict:
                self.logger.warning(f"No weight files found for {model_name}")
                self.logger.warning(f"Tried paths:")
                self.logger.warning(f"noprob: {noprob_pattern}")
                self.logger.warning(f"prob: {prob_pattern}")
            
        except Exception as e:
            self.logger.error(f"Error loading weights for {model_name}: {str(e)}")
            self.logger.error(f"Exception type: {type(e)}")
            self.logger.error(f"Stack trace:", exc_info=True)
        
        return weights_dict

    def get_result_path(self, subdir: str, filename: str) -> str:
        """결과 파일의 전체 경로를 반환합니다."""
        return os.path.join(self.result_dir, subdir, filename)

    def load_benchmark(self) -> pd.Series:
        """S&P 500 지수 데이터를 로드합니다."""
        try:
            # ImagePortOpt 프로젝트 루트 디렉토리 찾기
            project_root = "/home/indi/codespace/ImagePortOpt"  # 절대 경로 사용
            benchmark_path = os.path.join(project_root, 'Data', 'processed', 'snp500_index.csv')
            
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

    def save_weights(self, weights: pd.DataFrame, name: str):
        """
        포트폴리오 가중치를 저장합니다.
        
        Args:
            weights (pd.DataFrame): 저장할 가중치
            name (str): 벤치마크 이름
        """
        save_path = self.config['benchmarks'][name]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        weights.to_csv(save_path)
        self.logger.info(f"Saved weights for {name} to {save_path}") 