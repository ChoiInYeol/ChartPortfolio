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
from collections import defaultdict

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
        개별 주식 가중치를 0~20% 사이로 제한합니다.
        """
        # Softmax로 합이 1이 되도록 함
        raw_weights = nn.functional.softmax(self.weights, dim=0)
        
        # 0~0.2 사이로 클리핑
        clipped_weights = torch.clamp(raw_weights, min=0.0, max=0.2)
        
        # 다시 정규화하여 합이 1이 되도록 함
        normalized_weights = clipped_weights / clipped_weights.sum()
        
        return normalized_weights

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
                          lr: float = 0.01,
                          silent: bool = False,
                          patience: int = 50,
                          min_delta: float = 1e-4) -> np.ndarray:
        """
        포트폴리오를 최적화합니다.
        
        Args:
            returns (pd.DataFrame): 수익률 데이터
            method (str): 최적화 방법 ('max_sharpe', 'min_variance', 'min_cvar')
            epochs (int): 최적화 반복 횟수
            lr (float): 학습률
            silent (bool): 진행바 표시 여부
            patience (int): early stopping patience
            min_delta (float): 최소 개선값
            
        Returns:
            np.ndarray: 최적화된 포트폴리오 가중치
        """
        returns_tensor = self.to_tensor(returns)
        num_assets = returns_tensor.shape[1]
        
        model = PortfolioOptimizer(num_assets).to(self.device)
        optimizer = Adam(model.parameters(), lr=lr)
        
        best_loss = float('inf')
        best_weights = None
        patience_counter = 0
        
        iterator = range(epochs) if silent else tqdm(range(epochs), desc="Optimizing")
        
        for epoch in iterator:
            optimizer.zero_grad()
            weights = model()
            
            # 가중치 제약조건 검증
            if not torch.isclose(weights.sum(), torch.tensor(1.0, device=self.device), rtol=1e-3):
                self.logger.warning(f"Weights sum not equal to 1: {weights.sum().item()}")
                continue
            
            if (weights < 0).any() or (weights > 1).any():
                self.logger.warning("Weights outside [0,1] range detected")
                continue
            
            loss = self._calculate_loss(returns_tensor, weights, method)
            
            # Loss가 유효한지 확인
            if not torch.isfinite(loss):
                self.logger.warning(f"Invalid loss value: {loss.item()}")
                continue
                
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            
            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                best_weights = weights.detach().cpu().numpy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                if not silent:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # 최종 가중치 검증
        if best_weights is None:
            self.logger.warning("Optimization failed to find valid weights, using equal weights")
            best_weights = np.ones(num_assets) / num_assets
        else:
            # 가중치 정규화
            best_weights = np.clip(best_weights, 0, 1)
            best_weights = best_weights / best_weights.sum()
        
        return best_weights

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
        folder_name = os.path.join(base_folder, f'{model}{window_size}')
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
        folder_name = os.path.join(base_folder, f'{model}{window_size}')
        file_name = f'optimization_results_{optimization_method}.pkl'
        file_path = os.path.join(folder_name, file_name)
        
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                optimized_returns, portfolio_weights = pickle.load(f)
            self.logger.info(f"Optimization results loaded from {file_path}")
            return optimized_returns, portfolio_weights
        
        return None

    def create_top100_portfolio(self, 
                              up_prob: pd.DataFrame,
                              returns: pd.DataFrame) -> pd.DataFrame:
        """상위 100개 종목 동일가중 포트폴리오를 생성합니다."""
        weights = pd.DataFrame(0.0, index=up_prob.index, columns=returns.columns)
        
        for date in tqdm(up_prob.index, desc="Creating Top 100 portfolio"):
            prob_series = up_prob.loc[date]
            top100_stocks = prob_series.nlargest(100).index
            weights.loc[date, top100_stocks] = 1.0 / 100.0
        
        return weights

    def create_bottom100_portfolio(self, 
                                    up_prob: pd.DataFrame,
                                    returns: pd.DataFrame) -> pd.DataFrame:
        """하위 100개 종목 동일가중 포트폴리오를 생성합니다."""
        weights = pd.DataFrame(0.0, index=up_prob.index, columns=returns.columns)
        
        for date in tqdm(up_prob.index, desc="Creating Bottom 100 portfolio"):
            prob_series = up_prob.loc[date]
            bottom100_stocks = prob_series.nsmallest(100).index
            weights.loc[date, bottom100_stocks] = 1.0 / 100.0
        
        return weights

    def create_optimized_portfolio(self, 
                                 up_prob: pd.DataFrame,
                                 returns: pd.DataFrame,
                                 method: str = 'max_sharpe',
                                 n_stocks: int = 50) -> pd.DataFrame:
        """최적화된 포트폴리오를 생성합니다."""
        # 캐시 파일 경로 설정
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'optimized_weights_{method}_n{n_stocks}.pkl')
        
        # 캐시 파일이 존재하는 경우 로드
        if os.path.exists(cache_file):
            self.logger.info(f"Loading cached weights for {method} portfolio")
            with open(cache_file, 'rb') as f:
                weights = pickle.load(f)
            
            # 캐시된 가중치의 유효성 검증
            if (weights.index == up_prob.index).all() and set(weights.columns) == set(returns.columns):
                return weights
            else:
                self.logger.warning("Cached weights are outdated, recalculating...")
        
        # 새로운 가중치 계산
        weights = pd.DataFrame(0.0, index=up_prob.index, columns=returns.columns)
        
        for i, current_date in enumerate(tqdm(up_prob.index, desc=f"Creating {method} portfolio")):
            try:
                # 현재 날짜까지의 데이터만 사용
                available_returns = returns.loc[:current_date]
                current_prob = up_prob.loc[current_date]
                
                # 상위 n개 종목 선택
                top_stocks = current_prob.nlargest(n_stocks).index
                
                # 최소 필요 데이터 기간 확인 (60)
                if len(available_returns) < 60:
                    self.logger.warning(f"Insufficient data for date {current_date}, using equal weights")
                    weights.loc[current_date, top_stocks] = 1.0 / n_stocks
                    continue
                
                # 최근 60일간의 수익률 데이터만 사용
                hist_returns = available_returns.tail(60)[top_stocks]
                
                # 최적화 수행
                opt_weights = self.optimize_portfolio(
                    hist_returns,
                    method=method,
                    epochs=1000,
                    lr=0.01,
                    silent=True,
                    patience=50,
                    min_delta=1e-4
                )
                
                # 가중치 검증
                if np.isnan(opt_weights).any() or not np.isclose(opt_weights.sum(), 1.0, rtol=1e-3):
                    self.logger.warning(f"Invalid weights for date {current_date}, using equal weights")
                    weights.loc[current_date, top_stocks] = 1.0 / n_stocks
                else:
                    weights.loc[current_date, top_stocks] = opt_weights
                
                # 디버깅 정보 출력
                self.logger.debug(f"Date: {current_date}, Method: {method}")
                self.logger.debug(f"Weight sum: {opt_weights.sum():.4f}")
                self.logger.debug(f"Max weight: {opt_weights.max():.4f}")
                self.logger.debug(f"Min weight: {opt_weights.min():.4f}")
                
            except Exception as e:
                self.logger.error(f"Error optimizing for date {current_date}: {str(e)}")
                weights.loc[current_date, top_stocks] = 1.0 / n_stocks
        
        # 계산된 가중치 캐시에 저장
        with open(cache_file, 'wb') as f:
            pickle.dump(weights, f)
        self.logger.info(f"Saved weights to cache: {cache_file}")
        
        return weights

    def create_benchmark_portfolios(self, 
                                    up_prob: pd.DataFrame,
                                    returns: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """벤치마크 포트폴리오들을 생성합니다."""
        benchmark_weights = {}
        
        self.logger.info("Creating benchmark portfolios...")
        
        # 캐시 디렉토리 설정
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # 동일가중 포트폴리오 캐시 확인 및 생성
        top100_cache = os.path.join(cache_dir, 'top100_equal_weights.pkl')
        bottom100_cache = os.path.join(cache_dir, 'bottom100_equal_weights.pkl')
        
        if os.path.exists(top100_cache):
            self.logger.info("Loading cached Top 100 weights")
            with open(top100_cache, 'rb') as f:
                benchmark_weights['top100_equal'] = pickle.load(f)
        else:
            benchmark_weights['top100_equal'] = self.create_top100_portfolio(up_prob, returns)
            with open(top100_cache, 'wb') as f:
                pickle.dump(benchmark_weights['top100_equal'], f)
        
        if os.path.exists(bottom100_cache):
            self.logger.info("Loading cached Bottom 100 weights")
            with open(bottom100_cache, 'rb') as f:
                benchmark_weights['bottom100_equal'] = pickle.load(f)
        else:
            benchmark_weights['bottom100_equal'] = self.create_bottom100_portfolio(up_prob, returns)
            with open(bottom100_cache, 'wb') as f:
                pickle.dump(benchmark_weights['bottom100_equal'], f)
        
        # 최적화 포트폴리오
        for method in ['max_sharpe', 'min_variance', 'min_cvar']:
            portfolio_name = f'optimized_{method}'
            benchmark_weights[portfolio_name] = self.create_optimized_portfolio(
                up_prob=up_prob,
                returns=returns,
                method=method,
                n_stocks=50
            )
        
        return benchmark_weights