"""
포트폴리오 최적화를 위한 모듈입니다.
다양한 최적화 전략과 포트폴리오 최적화 기능을 포함합니다.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Union, Optional
import logging
from tqdm import tqdm
import os
import pickle

class PortfolioOptimizer(nn.Module):
    """
    포트폴리오 최적화를 위한 PyTorch 모델입니다.
    """
    
    def __init__(self, num_assets: int):
        """
        PortfolioOptimizer 초기화
        
        Args:
            num_assets (int): 자산 수
        """
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_assets) / num_assets)

    def forward(self) -> torch.Tensor:
        """
        포트폴리오 가중치를 계산합니다.
        
        Returns:
            torch.Tensor: 정규화된 포트폴리오 가중치
        """
        return nn.functional.softmax(self.weights, dim=0)

class OptimizationManager:
    """
    포트폴리오 최적화 프로세스를 관리하는 클래스입니다.
    """
    
    def __init__(self, device: torch.device):
        """
        OptimizationManager 초기화
        
        Args:
            device (torch.device): 계산에 사용할 디바이스 (CPU/GPU)
        """
        self.device = device
        self.logger = logging.getLogger(__name__)

    def to_tensor(self, data: Union[pd.DataFrame, np.ndarray]) -> torch.Tensor:
        """
        데이터를 PyTorch 텐서로 변환합니다.
        
        Args:
            data: 변환할 데이터
            
        Returns:
            torch.Tensor: 변환된 텐서
        """
        if isinstance(data, pd.DataFrame):
            return torch.tensor(data.values, dtype=torch.float32, device=self.device)
        elif isinstance(data, np.ndarray):
            return torch.tensor(data, dtype=torch.float32, device=self.device)
        return data

    def optimize_portfolio(self, 
                         returns: pd.DataFrame, 
                         method: str = 'max_sharpe', 
                         epochs: int = 1000, 
                         lr: float = 0.01) -> np.ndarray:
        """
        포트폴리오를 최적화합니다.
        
        Args:
            returns (pd.DataFrame): 수익률 데이터
            method (str): 최적화 방법 ('max_sharpe', 'min_variance', 'min_cvar')
            epochs (int): 최적화 반복 횟수
            lr (float): 학습률
            
        Returns:
            np.ndarray: 최적화된 포트폴리오 가중치
        """
        self.logger.info(f"Optimizing portfolio using {method} method")
        returns_tensor = self.to_tensor(returns)
        num_assets = returns_tensor.shape[1]
        
        model = PortfolioOptimizer(num_assets).to(self.device)
        optimizer = Adam(model.parameters(), lr=lr)
        
        for _ in tqdm(range(epochs), desc="Optimizing"):
            optimizer.zero_grad()
            weights = model()
            
            loss = self._calculate_loss(returns_tensor, weights, method)
            loss.backward()
            optimizer.step()
        
        optimized_weights = model().detach().cpu().numpy()
        self.logger.info(f"Optimization completed - Weights shape: {optimized_weights.shape}")
        return optimized_weights

    def _calculate_loss(self, 
                       returns: torch.Tensor, 
                       weights: torch.Tensor, 
                       method: str) -> torch.Tensor:
        """
        최적화 목적함수의 손실을 계산합니다.
        
        Args:
            returns (torch.Tensor): 수익률 텐서
            weights (torch.Tensor): 포트폴리오 가중치
            method (str): 최적화 방법
            
        Returns:
            torch.Tensor: 계산된 손실값
        """
        if method == 'max_sharpe':
            portfolio_return = (returns.mean(dim=0) * weights).sum()
            portfolio_volatility = torch.sqrt(
                torch.dot(weights, torch.matmul(returns.T, returns).matmul(weights))
            )
            sharpe_ratio = portfolio_return / portfolio_volatility
            return -sharpe_ratio
            
        elif method == 'min_variance':
            portfolio_volatility = torch.sqrt(
                torch.dot(weights, torch.matmul(returns.T, returns).matmul(weights))
            )
            return portfolio_volatility
            
        elif method == 'min_cvar':
            portfolio_returns = torch.matmul(returns, weights)
            var = torch.quantile(portfolio_returns, 0.05)
            cvar = portfolio_returns[portfolio_returns <= var].mean()
            return -cvar
            
        else:
            raise ValueError("Invalid optimization method")

    def process_portfolio(self,
                        ensemble_results: pd.DataFrame,
                        us_ret: pd.DataFrame,
                        method: str,
                        model: str,
                        model_window_size: int,
                        base_folder: str,
                        optimization_window_size: int = 60) -> Tuple[pd.Series, Dict]:
        """
        포트폴리오 최적화를 수행합니다.
        
        Args:
            ensemble_results (pd.DataFrame): 앙상블 결과
            us_ret (pd.DataFrame): 수익률 데이터
            method (str): 최적화 방법
            model (str): 모델 이름
            model_window_size (int): 모델 윈도우 크기
            base_folder (str): 기본 폴더 경로
            optimization_window_size (int): 최적화 윈도우 크기
            
        Returns:
            Tuple[pd.Series, Dict]: 최적화된 수익률과 포트폴리오 가중치
        """
        # 저장된 결과가 있는지 확인
        saved_results = self._load_optimization_results(
            model, model_window_size, method, base_folder
        )
        if saved_results is not None:
            return saved_results
            
        # TOP 100 종목 선택
        selected_stocks = ensemble_results.sort_values(
            ['investment_date', 'up_prob'], 
            ascending=[True, False]
        ).groupby('investment_date').head(100)
        
        optimized_returns = []
        portfolio_weights = {}
        
        rebalance_dates = selected_stocks['investment_date'].unique()
        
        for i, current_date in enumerate(tqdm(rebalance_dates, desc="Processing dates")):
            try:
                results = self._process_single_period(
                    current_date,
                    i,
                    rebalance_dates,
                    us_ret,
                    selected_stocks,  # TOP 100 종목만 포함된 데이터
                    method,
                    model_window_size,
                    optimization_window_size
                )
                if results is not None:
                    opt_returns, weights = results
                    optimized_returns.extend(opt_returns)
                    portfolio_weights[current_date] = weights
            except Exception as e:
                self.logger.error(f"Error in portfolio optimization for date {current_date}: {str(e)}")
        
        optimized_returns = pd.Series(dict(optimized_returns))
        
        # 결과 저장
        self._save_optimization_results(
            optimized_returns, portfolio_weights, model, model_window_size, method, base_folder
        )
        
        return optimized_returns, portfolio_weights

    def _process_single_period(self,
                            current_date: pd.Timestamp,
                            i: int,
                            rebalance_dates: np.ndarray,
                            us_ret: pd.DataFrame,
                            selected_stocks: pd.DataFrame,
                            method: str,
                            model_window_size: int,
                            optimization_window_size: int) -> Optional[Tuple[list, dict]]:
        """
        단일 기간의 포트폴리오 최적화를 처리합니다.
        """
        # 현재 리밸런싱 날짜 이후의 첫 거래일 찾기
        available_dates = us_ret.index
        future_dates = available_dates[available_dates >= current_date]
        if len(future_dates) == 0:
            return None
        
        current_date = future_dates[0]
        current_index = us_ret.index.get_loc(current_date)
        
        # 과거 optimization_window_size 기간의 거래일 데이터 가져오기
        start_index = max(0, current_index - optimization_window_size)
        historical_returns = us_ret.iloc[start_index:current_index]
        
        selected_stocks_for_date = selected_stocks[
            selected_stocks['investment_date'] == rebalance_dates[i]
        ]['StockID']
        historical_returns = historical_returns[selected_stocks_for_date]
        
        if historical_returns.empty or historical_returns.isnull().all().all():
            return None

        # 최적화된 가중치 계산
        weights = self.optimize_portfolio(historical_returns, method=method)
        
        # 포트폴리오 가중치 저장 (선택된 종목들에 대해서만)
        portfolio_weights = dict(zip(us_ret.columns, np.zeros(len(us_ret.columns))))
        for stock, weight in zip(historical_returns.columns, weights):
            portfolio_weights[stock] = weight
        
        # 다음 리밸런싱 날짜의 첫 거래일 찾기
        if i < len(rebalance_dates) - 1:
            next_rebalance = rebalance_dates[i + 1]
            future_dates = available_dates[available_dates >= next_rebalance]
            if len(future_dates) == 0:
                next_rebalance_date = available_dates[-1]
            else:
                next_rebalance_date = future_dates[0]
        else:
            next_rebalance_date = available_dates[-1]
        
        next_period_returns = us_ret.loc[current_date:next_rebalance_date, historical_returns.columns]
        optimized_return = (next_period_returns * weights).sum(axis=1)
        
        return list(zip(next_period_returns.index, optimized_return)), portfolio_weights

    def _save_optimization_results(self,
                                 optimized_returns: pd.Series,
                                 portfolio_weights: Dict,
                                 model: str,
                                 window_size: int,
                                 optimization_method: str,
                                 base_folder: str):
        """
        최적화 결과를 저장합니다.
        """
        folder_name = os.path.join(base_folder, 'WORK_DIR', f'{model}{window_size}')
        os.makedirs(folder_name, exist_ok=True)
        file_name = f'optimization_results_{optimization_method}.pkl'
        file_path = os.path.join(folder_name, file_name)
        
        with open(file_path, 'wb') as f:
            pickle.dump((optimized_returns, portfolio_weights), f)
        
        self.logger.info(f"Optimization results saved to {file_path}")

    def _load_optimization_results(self,
                                 model: str,
                                 window_size: int,
                                 optimization_method: str,
                                 base_folder: str) -> Optional[Tuple[pd.Series, Dict]]:
        """
        저장된 최적화 결과를 로드합니다.
        """
        folder_name = os.path.join(base_folder, 'WORK_DIR', f'{model}{window_size}')
        file_name = f'optimization_results_{optimization_method}.pkl'
        file_path = os.path.join(folder_name, file_name)
        
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                optimized_returns, portfolio_weights = pickle.load(f)
            self.logger.info(f"Optimization results loaded from {file_path}")
            return optimized_returns, portfolio_weights
        
        return None