"""포트폴리오 최적화를 위한 모듈입니다."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging
import torch
from tqdm import tqdm
import os


class OptimizationManager:
    """포트폴리오 최적화를 위한 클래스입니다."""
    
    def __init__(self):
        """OptimizationManager 초기화"""
        self.logger = logging.getLogger(__name__)

    def optimize_portfolio_with_torch(self, 
                                      returns: pd.DataFrame,
                                      method: str = 'max_sharpe',
                                      constraints: Dict = None,
                                      show_progress: bool = True) -> np.ndarray:
        """PyTorch를 사용하여 포트폴리오를 최적화합니다."""
        try:
            n_assets = returns.shape[1]
            
            # 시드 고정
            torch.manual_seed(42)
            np.random.seed(42)
            
            # 데이터 검증
            if n_assets < 2:
                self.logger.warning("Not enough assets for optimization")
                return np.ones(n_assets) / n_assets
            
            if len(returns) < 20:
                self.logger.warning("Not enough historical data for optimization")
                return np.ones(n_assets) / n_assets
            
            # 기본 제약조건 설정
            if constraints is None:
                constraints = {
                    'min_weight': 0.0,
                    'max_weight': 0.2
                }
            
            # 여러 번의 시도를 통한 최적화
            best_weights = None
            best_objective = float('inf')
            n_attempts = 5  # 시도 횟수
            
            for attempt in range(n_attempts):
                try:
                    # 초기 가중치 설정 (각 시도마다 다른 초기값)
                    initial_weights = torch.from_numpy(
                        np.random.dirichlet(np.ones(n_assets) * (attempt + 1))
                    ).float().cuda()
                    
                    # 공분산 행렬 계산 및 안정화
                    returns_tensor = torch.tensor(returns.values, dtype=torch.float32, device='cuda')
                    cov_matrix = torch.cov(returns_tensor.T) * 252
                    
                    # 공분산 행렬 안정화 강화
                    stability_factor = 1e-6 * (attempt + 1)  # 각 시도마다 다른 안정화 계수
                    cov_matrix += torch.eye(n_assets, device='cuda') * stability_factor
                    
                    # 목적함수 설정
                    if method == 'max_sharpe':
                        def objective(weights):
                            return -self._sharpe_ratio_torch(weights, returns_tensor)
                    elif method == 'min_variance':
                        def objective(weights):
                            variance = self._portfolio_variance_torch(weights, cov_matrix)
                            # 분산이 너무 작은 경우 페널티 추가
                            if variance < 1e-10:
                                return torch.tensor(float('inf'), device='cuda')
                            return variance
                    elif method == 'min_cvar':
                        def objective(weights):
                            return -self._portfolio_cvar_torch(weights, returns_tensor)
                    
                    # 최적화 실행
                    weights = initial_weights.clone().requires_grad_(True)
                    optimizer = torch.optim.Adam([weights], lr=0.01 / (attempt + 1))  # 학습률 조정
                    
                    for epoch in range(300):
                        optimizer.zero_grad()
                        loss = objective(weights)
                        
                        if not torch.isnan(loss) and not torch.isinf(loss):
                            loss.backward()
                            optimizer.step()
                            
                            with torch.no_grad():
                                weights.clamp_(constraints['min_weight'], constraints['max_weight'])
                                weights /= weights.sum()
                    
                    # 최적의 가중치 업데이트
                    final_loss = objective(weights).item()
                    if final_loss < best_objective and not np.isnan(final_loss):
                        best_objective = final_loss
                        best_weights = weights.detach().cpu().numpy()
                
                except Exception as e:
                    self.logger.warning(f"Optimization attempt {attempt + 1} failed: {str(e)}")
                    continue
            
            # 모든 시도가 실패한 경우 동일가중치 반환
            if best_weights is None:
                self.logger.warning("All optimization attempts failed, using equal weights")
                return np.ones(n_assets) / n_assets
            
            # 최종 가중치 정규화
            best_weights = best_weights / np.sum(best_weights)
            
            # 가중치 검증
            if np.all(best_weights == 0) or np.any(np.isnan(best_weights)):
                self.logger.warning("Invalid weights detected, using equal weights")
                return np.ones(n_assets) / n_assets
            
            return best_weights
            
        except Exception as e:
            self.logger.error(f"Error during optimization: {str(e)}")
            return np.ones(n_assets) / n_assets

    def _sharpe_ratio_torch(self, weights: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """PyTorch를 사용하여 Sharpe ratio를 계산합니다."""
        portfolio_ret = torch.mean(returns @ weights) * 252
        portfolio_vol = torch.sqrt(weights @ torch.cov(returns.T) * 252 @ weights)
        
        if portfolio_vol == 0 or torch.isnan(portfolio_vol) or torch.isinf(portfolio_vol):
            return torch.tensor(0.0, device='cuda')
        
        return portfolio_ret / portfolio_vol

    def _portfolio_variance_torch(self, weights: torch.Tensor, cov_matrix: torch.Tensor) -> torch.Tensor:
        """PyTorch를 사용하여 포트폴리오 분산을 계산합니다."""
        return weights @ cov_matrix @ weights

    def _portfolio_cvar_torch(self, weights: torch.Tensor, returns: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
        """PyTorch를 사용하여 포트폴리오 CVaR을 계산합니다."""
        portfolio_returns = returns @ weights
        var = torch.quantile(portfolio_returns, alpha)
        below_var = portfolio_returns[portfolio_returns <= var]
        
        if len(below_var) == 0:
            return torch.tensor(float('inf'), device='cuda')
        
        cvar = torch.mean(below_var)
        
        if torch.isnan(cvar) or torch.isinf(cvar):
            return torch.tensor(float('inf'), device='cuda')
        
        return cvar

    def create_benchmark_portfolios(self, 
                                  up_prob: pd.DataFrame,
                                  returns: pd.DataFrame,
                                  N: int = None,
                                  result_dir: str = 'results',
                                  rebalance_dates: pd.DatetimeIndex = None) -> Dict[str, pd.DataFrame]:
        """
        벤치마크 포트폴리오들을 생성합니다.
        
        Args:
            up_prob (pd.DataFrame): 상승확률 데이터
            returns (pd.DataFrame): 수익률 데이터
            N (int, optional): 선택할 종목 수
            result_dir (str): 결과 저장 경로
            rebalance_dates (pd.DatetimeIndex): 리밸런싱 날짜
        
        Returns:
            Dict[str, pd.DataFrame]: 포트폴리오별 가중치
        """
        benchmark_weights = {}
        lookback_period = 60
        
        # N개 종목 선택
        top_N_columns = returns.columns[:N] if N else returns.columns
        
        # 첫 번째 가능한 날짜 계산
        first_valid_date = returns.index[lookback_period]
        
        # 리밸런싱 날짜가 제공되지 않은 경우
        if rebalance_dates is None:
            rebalance_dates = pd.date_range(
                start=first_valid_date,
                end=returns.index[-1],
                freq='ME'
            )
        
        # 각 리밸런싱 날짜에 대해 처리
        for date in tqdm(rebalance_dates[rebalance_dates >= first_valid_date],
                        desc="Creating benchmark portfolios"):
            # 해당 월의 상승확률 데이터 선택
            month_probs = up_prob.loc[date]
            if N is not None:
                month_probs = month_probs[top_N_columns]
            
            # 상위 50% 종목 선택
            threshold = month_probs.median()
            up_stocks = month_probs[month_probs >= threshold].index
            
            # CNN Top 포트폴리오 초기화
            if 'CNN Top' not in benchmark_weights:
                benchmark_weights['CNN Top'] = pd.DataFrame(
                    0.0,
                    index=rebalance_dates,
                    columns=returns.columns,
                    dtype=np.float64
                )
            
            # 상승예측 종목이 없는 경우 처리
            if len(up_stocks) == 0:
                # 이전 달의 가중치를 유지
                prev_date = benchmark_weights['CNN Top'].index[
                    benchmark_weights['CNN Top'].index < date
                ][-1]
                for portfolio_name in benchmark_weights:
                    benchmark_weights[portfolio_name].loc[date] = \
                        benchmark_weights[portfolio_name].loc[prev_date]
                continue
            
            # CNN Top (상승예측 종목 동일가중)
            equal_weight = np.float64(1.0/len(up_stocks))
            benchmark_weights['CNN Top'].loc[date, up_stocks] = equal_weight
            
            # 최적화 포트폴리오 생성
            for method in ['max_sharpe', 'min_variance', 'min_cvar']:
                method_name = method.replace('_', ' ').title()
                if method == 'min_cvar':
                    method_name = 'Min CVaR'
                
                # 기본 포트폴리오 초기화
                if method_name not in benchmark_weights:
                    benchmark_weights[method_name] = pd.DataFrame(
                        0.0,
                        index=rebalance_dates,
                        columns=returns.columns,
                        dtype=np.float64
                    )
                
                try:
                    # 전체 종목에 대한 최적화
                    weights = self.optimize_portfolio_with_torch(
                        returns.loc[:date].tail(lookback_period), 
                        method=method,
                        show_progress=False
                    )
                    weights = np.array(weights, dtype=np.float64)
                    
                    if N is not None:
                        benchmark_weights[method_name].loc[date, top_N_columns] = weights
                    else:
                        benchmark_weights[method_name].loc[date] = weights
                    
                except Exception as e:
                    self.logger.error(f"Error optimizing {method_name} for date {date}: {str(e)}")
                    # 최적화 실패 시 이전 가중치 유지
                    prev_date = benchmark_weights[method_name].index[
                        benchmark_weights[method_name].index < date
                    ][-1]
                    benchmark_weights[method_name].loc[date] = \
                        benchmark_weights[method_name].loc[prev_date]
                
                # CNN Top + 최적화 포트폴리오
                portfolio_name = f'CNN Top + {method_name}'
                if portfolio_name not in benchmark_weights:
                    benchmark_weights[portfolio_name] = pd.DataFrame(
                        0.0,
                        index=rebalance_dates,
                        columns=returns.columns,
                        dtype=np.float64
                    )
                
                try:
                    # CNN이 선택한 종목들에 대해서만 최적화
                    up_returns = returns.loc[:date].tail(lookback_period)[up_stocks]
                    weights = self.optimize_portfolio_with_torch(
                        up_returns,
                        method=method,
                        show_progress=False
                    )
                    weights = np.array(weights, dtype=np.float64)
                    benchmark_weights[portfolio_name].loc[date, up_stocks] = weights
                    
                except Exception as e:
                    self.logger.error(f"Error optimizing {portfolio_name} for date {date}: {str(e)}")
                    # 최적화 실패 시 동일가중
                    benchmark_weights[portfolio_name].loc[date, up_stocks] = equal_weight
        
        return benchmark_weights