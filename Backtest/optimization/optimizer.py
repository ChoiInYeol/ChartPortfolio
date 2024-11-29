"""포트폴리오 최적화를 위한 모듈입니다."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging
from scipy.optimize import minimize

class OptimizationManager:
    """포트폴리오 최적화를 위한 클래스입니다."""
    
    def __init__(self):
        """OptimizationManager 초기화"""
        self.logger = logging.getLogger(__name__)

    def optimize_portfolio(self, 
                         returns: pd.DataFrame,
                         method: str = 'max_sharpe',
                         constraints: Dict = None) -> np.ndarray:
        """포트폴리오를 최적화합니다."""
        try:
            n_assets = returns.shape[1]
            
            # 데이터 검증
            if n_assets < 2:
                self.logger.warning("Not enough assets for optimization")
                return np.ones(n_assets) / n_assets
            
            if len(returns) < 20:  # 최소 20일의 데이터 필요
                self.logger.warning("Not enough historical data for optimization")
                return np.ones(n_assets) / n_assets
            
            # 기본 제약조건 설정
            if constraints is None:
                constraints = {
                    'min_weight': 0.0,
                    'max_weight': 0.2
                }
            
            # 초기 가중치 설정 (동일가중)
            initial_weights = np.ones(n_assets) / n_assets
            
            # 공분산 행렬 계산
            returns_np = returns.values  # numpy array로 변환
            cov_matrix = np.cov(returns_np.T) * 252
            
            # 공분산 행렬 유효성 검사
            if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any():
                self.logger.warning("Invalid covariance matrix")
                return initial_weights
            
            # 목적함수 설정
            if method == 'max_sharpe':
                objective = lambda w: -self._sharpe_ratio(w, returns_np)
            elif method == 'min_variance':
                objective = lambda w: self._portfolio_variance(w, cov_matrix)
            elif method == 'min_cvar':
                objective = lambda w: -self._portfolio_cvar(w, returns_np)
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            # 제약조건 정의
            constraints_list = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'ineq', 'fun': lambda x: x - constraints['min_weight']},
                {'type': 'ineq', 'fun': lambda x: constraints['max_weight'] - x}
            ]
            
            bounds = [(constraints['min_weight'], constraints['max_weight'])] * n_assets
            
            # 최적화 실행
            result = minimize(
                fun=objective,
                x0=initial_weights,
                method='SLSQP',
                constraints=constraints_list,
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            
            if not result.success:
                self.logger.warning(f"Optimization failed: {result.message}")
                return initial_weights
            
            # 결과 검증
            weights = result.x
            if np.isnan(weights).any() or np.isinf(weights).any():
                self.logger.warning("Invalid optimization result")
                return initial_weights
            
            # 가중치 합이 1이 되도록 정규화
            weights = weights / np.sum(weights)
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Error during optimization: {str(e)}")
            return np.ones(n_assets) / n_assets

    def _sharpe_ratio(self, weights: np.ndarray, returns: np.ndarray) -> float:
        """Sharpe ratio를 계산합니다."""
        try:
            portfolio_ret = np.mean(returns @ weights) * 252
            portfolio_vol = np.sqrt(weights @ np.cov(returns.T) * 252 @ weights)
            
            if portfolio_vol == 0 or np.isnan(portfolio_vol) or np.isinf(portfolio_vol):
                return 0
            
            return portfolio_ret / portfolio_vol
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0

    def _portfolio_variance(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """포트폴리오 분산을 계산합니다."""
        try:
            return weights @ cov_matrix @ weights
        except Exception as e:
            self.logger.error(f"Error calculating portfolio variance: {str(e)}")
            return np.inf

    def _portfolio_cvar(self, weights: np.ndarray, returns: np.ndarray, alpha: float = 0.05) -> float:
        """포트폴리오 CVaR을 계산합니다."""
        try:
            portfolio_returns = returns @ weights
            var = np.percentile(portfolio_returns, alpha * 100)
            below_var = portfolio_returns[portfolio_returns <= var]
            
            if len(below_var) == 0:
                return np.inf
            
            cvar = np.mean(below_var)
            
            if np.isnan(cvar) or np.isinf(cvar):
                return np.inf
            
            return cvar
            
        except Exception as e:
            self.logger.error(f"Error calculating CVaR: {str(e)}")
            return np.inf

    def create_benchmark_portfolios(self, 
                                  up_prob: pd.DataFrame,
                                  returns: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """벤치마크 포트폴리오들을 생성합니다."""
        benchmark_weights = {}
        
        # 1. Naive (Universe) - 시가총액 상위 50종목 동일가중
        top_50_columns = returns.columns[:50]
        naive_weights = pd.DataFrame(0.0, 
                                   index=returns.index,
                                   columns=returns.columns,
                                   dtype=np.float64)  # dtype 명시
        naive_weights.loc[:, top_50_columns] = 1.0/50
        benchmark_weights['Naive'] = naive_weights
        
        # 각 시점별로 처리
        for date in up_prob.index:
            period_returns = returns.loc[:date].tail(60)
            
            # 2. CNN 기반 포트폴리오
            probs = up_prob.loc[date]
            up_stocks = probs[probs >= 0.5].index
            down_stocks = probs[probs < 0.5].index
            
            # CNN Top (상승예측 종목 동일가중)
            if 'CNN Top' not in benchmark_weights:
                benchmark_weights['CNN Top'] = pd.DataFrame(0.0,  # 0.0으로 초기화
                                                          index=up_prob.index,
                                                          columns=returns.columns,
                                                          dtype=np.float64)  # dtype 명시
            if len(up_stocks) > 0:
                equal_weight = np.float64(1.0/len(up_stocks))  # dtype 명시
                benchmark_weights['CNN Top'].loc[date, up_stocks] = equal_weight
            
            # # CNN Bottom (하락예측 종목 동일가중)
            # if 'CNN Bottom' not in benchmark_weights:
            #     benchmark_weights['CNN Bottom'] = pd.DataFrame(0.0,
            #                                                  index=up_prob.index,
            #                                                  columns=returns.columns,
            #                                                  dtype=np.float64)
            # if len(down_stocks) > 0:
            #     equal_weight = np.float64(1.0/len(down_stocks))
            #     benchmark_weights['CNN Bottom'].loc[date, down_stocks] = equal_weight
            
            # 3. 최적화 포트폴리오
            # Universe 전체에 대한 최적화
            for method in ['max_sharpe', 'min_variance', 'min_cvar']:
                portfolio_name = method.replace('_', ' ').title()
                if portfolio_name not in benchmark_weights:
                    benchmark_weights[portfolio_name] = pd.DataFrame(0.0,
                                                                  index=up_prob.index,
                                                                  columns=returns.columns,
                                                                  dtype=np.float64)
                
                weights = self.optimize_portfolio(period_returns, method=method)
                weights = np.array(weights, dtype=np.float64)  # dtype 명시
                benchmark_weights[portfolio_name].loc[date] = weights
            
            # CNN Top 종목들에 대한 최적화
            if len(up_stocks) > 0:
                up_returns = period_returns[up_stocks]
                for method in ['max_sharpe', 'min_variance', 'min_cvar']:
                    portfolio_name = f'CNN Top + {method.replace("_", " ").title()}'
                    if portfolio_name not in benchmark_weights:
                        benchmark_weights[portfolio_name] = pd.DataFrame(0.0,
                                                                      index=up_prob.index,
                                                                      columns=returns.columns,
                                                                      dtype=np.float64)
                    
                    weights = self.optimize_portfolio(up_returns, method=method)
                    weights = np.array(weights, dtype=np.float64)
                    benchmark_weights[portfolio_name].loc[date, up_stocks] = weights
        
        return benchmark_weights