"""
포트폴리오 성과 측정을 위한 모듈입니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
import os

class PerformanceMetrics:
    """
    포트폴리오 성과 지표를 계산하는 클래스입니다.
    """
    
    def __init__(self):
        """
        PerformanceMetrics 초기화
        """
        self.logger = logging.getLogger(__name__)

    def calculate_portfolio_metrics(self, 
                                  returns: pd.Series, 
                                  weights: Optional[pd.DataFrame] = None,
                                  benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        포트폴리오 성과 지표를 계산합니다.
        
        Args:
            returns (pd.Series): 수익률 시계열
            weights (pd.DataFrame, optional): 포트폴리오 가중치
            benchmark_returns (pd.Series, optional): 벤치마크 수익률
            
        Returns:
            Dict[str, float]: 계산된 성과 지표
        """
        # 연율화 팩터
        annual_factor = 252
        risk_free_rate = 0.02  # 연 2% 가정
        
        # 1. Expected Return (연율화)
        mean_return = returns.mean() * annual_factor
        
        # 2. Standard Deviation (연율화)
        std_return = returns.std() * np.sqrt(annual_factor)
        
        # 3. Sharpe Ratio
        sharpe_ratio = (mean_return - risk_free_rate) / std_return if std_return != 0 else 0
        
        # 4. Downside Deviation
        downside_returns = returns[returns < 0]
        downside_dev = downside_returns.std() * np.sqrt(annual_factor) if len(downside_returns) > 0 else 0
        
        # 5. Sortino Ratio
        sortino_ratio = (mean_return - risk_free_rate) / downside_dev if downside_dev != 0 else 0
        
        # 6. Maximum Drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # 7. Percentage of Positive Returns
        pos_return_ratio = (returns > 0).mean()
        
        # 8. Turnover (가중치가 제공된 경우)
        turnover = 0.0
        if weights is not None:
            weight_changes = weights.diff().abs().sum(axis=1)
            turnover = weight_changes.mean()
        
        # 9. Beta (벤치마크가 제공된 경우)
        beta = np.nan
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            covariance = returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance != 0 else np.nan
        
        return {
            'E(R)': mean_return,
            'Std(R)': std_return,
            'Sharpe': sharpe_ratio,
            'DD(R)': downside_dev,
            'Sortino': sortino_ratio,
            'MDD': max_drawdown,
            '% of +Ret': pos_return_ratio,
            'Turnover': turnover,
            'Beta': beta
        }

    def save_metrics_latex(self, 
                          metrics: pd.DataFrame,
                          result_dir: str,
                          model_name: str) -> None:
        """성과 지표를 LaTeX 형식으로 저장합니다."""
        latex_path = os.path.join(result_dir, f'metrics_{model_name}.tex')
        
        with open(latex_path, 'w') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Portfolio Performance Metrics}\n")
            f.write("\\begin{tabular}{l" + "r" * len(metrics.columns) + "}\n")
            f.write("\\hline\n")
            f.write("Portfolio & " + " & ".join(metrics.index) + " \\\\\n")
            f.write("\\hline\n")
            
            for col in metrics.columns:
                row = [col]
                for idx in metrics.index:
                    value = metrics.loc[idx, col]
                    row.append(f"{value:.4f}")
                f.write(" & ".join(row) + " \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}")
        
        # CSV 형식으로도 저장
        csv_path = os.path.join(result_dir, f'metrics_{model_name}.csv')
        metrics.to_csv(csv_path, float_format='%.4f')