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
        """
        포트폴리오를 최적화합니다.
        
        Args:
            returns (pd.DataFrame): 수익률 데이터
            method (str): 최적화 방법 ('max_sharpe', 'min_variance', 'min_cvar')
            constraints (Dict): 제약조건 (예: {'min_weight': 0, 'max_weight': 0.2})
            
        Returns:
            np.ndarray: 최적화된 포트폴리오 가중치
        """
        n_assets = returns.shape[1]
        
        # 기본 제약조건 설정
        if constraints is None:
            constraints = {
                'min_weight': 0.0,
                'max_weight': 0.2  # 개별 종목 최대 20%
            }
        
        # 목적함수 및 제약조건 설정
        if method == 'max_sharpe':
            objective = self._negative_sharpe_ratio
        elif method == 'min_variance':
            objective = self._portfolio_variance
        elif method == 'min_cvar':
            objective = self._portfolio_cvar
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # 초기 가중치 설정 (동일가중)
        initial_weights = np.ones(n_assets) / n_assets
        
        # 제약조건 정의
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 가중치 합 = 1
            {'type': 'ineq', 'fun': lambda x: x - constraints['min_weight']},  # 최소 가중치
            {'type': 'ineq', 'fun': lambda x: constraints['max_weight'] - x}   # 최대 가중치
        ]
        
        # 최적화 실행
        try:
            result = minimize(
                fun=lambda w: objective(w, returns),
                x0=initial_weights,
                method='SLSQP',
                constraints=constraints_list,
                bounds=[(constraints['min_weight'], constraints['max_weight'])] * n_assets
            )
            
            if not result.success:
                self.logger.warning(f"Optimization failed: {result.message}")
                return initial_weights
                
            return result.x
            
        except Exception as e:
            self.logger.error(f"Error during optimization: {str(e)}")
            return initial_weights

    def _negative_sharpe_ratio(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        """Sharpe ratio의 음수값을 계산합니다."""
        portfolio_ret = np.sum(returns.mean() * weights) * 252
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        return -portfolio_ret / portfolio_vol if portfolio_vol != 0 else 0

    def _portfolio_variance(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        """포트폴리오 분산을 계산합니다."""
        return np.dot(weights.T, np.dot(returns.cov() * 252, weights))

    def _portfolio_cvar(self, weights: np.ndarray, returns: pd.DataFrame, alpha: float = 0.05) -> float:
        """포트폴리오 CVaR을 계산합니다."""
        portfolio_returns = np.dot(returns, weights)
        var = np.percentile(portfolio_returns, alpha * 100)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return -cvar if not np.isnan(cvar) else np.inf

    def create_benchmark_portfolios(self, 
                                  up_prob: pd.DataFrame,
                                  returns: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        벤치마크 포트폴리오들을 생성합니다.
        
        Args:
            up_prob (pd.DataFrame): 상승확률 데이터
            returns (pd.DataFrame): 수익률 데이터
            
        Returns:
            Dict[str, pd.DataFrame]: 포트폴리오별 가중치
        """
        benchmark_weights = {}
        
        # 각 시점별로 처리
        for date in up_prob.index:
            # Top 50 종목 선택
            top_stocks = up_prob.loc[date].nlargest(50).index
            period_returns = returns.loc[:date, top_stocks].tail(60)  # 최근 60일 데이터 사용
            
            # 각 최적화 방법별로 가중치 계산
            for method in ['max_sharpe', 'min_variance', 'min_cvar']:
                if method not in benchmark_weights:
                    benchmark_weights[f'CNN Top 50 + {method.replace("_", " ").title()}'] = \
                        pd.DataFrame(0, index=up_prob.index, columns=returns.columns)
                
                weights = self.optimize_portfolio(period_returns, method=method)
                benchmark_weights[f'CNN Top 50 + {method.replace("_", " ").title()}'].loc[date, top_stocks] = weights
        
        return benchmark_weights