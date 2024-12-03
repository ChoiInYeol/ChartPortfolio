"""데이터 로딩을 위한 모듈입니다."""

import pandas as pd
import numpy as np
import logging
import os
import yaml
from typing import Dict

class DataLoader:
    """데이터 로딩을 위한 클래스입니다."""
    
    def __init__(self, base_folder: str, data_size: int, train_date: str = '2017-12-31', end_date: str = '2024-07-05'):
        """
        DataLoader 초기화
        
        Args:
            base_folder (str): 기본 폴더 경로
            data_size (int): 데이터 크기 (50, 370, 500, 2055)
            train_date (str): 학습 시작 날짜
            end_date (str): 투자 종료 날짜
        """
        self.base_folder = base_folder
        self.data_size = data_size
        self.train_date = train_date
        self.end_date = end_date
        self.logger = logging.getLogger(__name__)
        
        # 설정 파일 로드
        with open(os.path.join(base_folder, 'weight.yaml'), 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 결과 디렉토리 설정
        self.result_dir = os.path.join(
            self.config['base_settings']['result_dir'],
            f'Result_{data_size}'
        )
        
        # 하위 디렉토리 생성
        for subdir in self.config['base_settings']['subdirs']:
            os.makedirs(os.path.join(self.result_dir, subdir), exist_ok=True)

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """필요한 데이터를 모두 로드합니다."""
        try:
            # 수익률 데이터 로드
            returns = pd.read_csv(f'../TS_Model/data/filtered_returns_top{self.data_size}.csv', 
                                index_col=0, parse_dates=True)
            
            # 상승확률 데이터 로드
            probs = pd.read_csv(f'../TS_Model/data/filtered_probs_top{self.data_size}.csv', 
                              index_col=0, parse_dates=True)
            
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
            model_templates = self.config['file_templates']['ts_models'].get(model_name, {})
            if not model_templates:
                self.logger.warning(f"No templates configured for model {model_name}")
                return weights_dict
            
            # 파일 경로 생성
            for weight_type, template in model_templates.items():
                path = template.format(size=self.data_size)
                if os.path.exists(path):
                    weights = pd.read_csv(path, index_col=0, parse_dates=True)
                    weights = weights[(weights.index >= self.train_date) & 
                                    (weights.index <= self.end_date)].fillna(0)
                    
                    if weight_type == 'noprob':
                        weights_dict[model_name] = weights
                    else:  # prob
                        weights_dict[f'CNN + {model_name}'] = weights
                        
        except Exception as e:
            self.logger.error(f"Error loading weights for {model_name}: {str(e)}")
        
        return weights_dict

    def get_result_path(self, subdir: str, filename: str) -> str:
        """결과 파일의 전체 경로를 반환합니다."""
        return os.path.join(self.result_dir, subdir, filename)

    def load_benchmark(self) -> pd.Series:
        """S&P 500 지수 데이터를 로드합니다."""
        try:
            benchmark = pd.read_csv('../Data/processed/snp500_index.csv', 
                                  index_col=0, parse_dates=True)
            
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