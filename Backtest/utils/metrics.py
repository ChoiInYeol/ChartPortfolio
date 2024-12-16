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

    def calculate_portfolio_returns(self, 
                                  returns: pd.DataFrame,
                                  weights: pd.DataFrame,
                                  rebalancing_freq: str = 'ME') -> pd.Series:
        """
        포트폴리오 수익률을 계산합니다.
        
        Args:
            returns (pd.DataFrame): 일별 수익률
            weights (pd.DataFrame): 포트폴리오 가중치
            rebalancing_freq (str): 리밸런싱 주기 ('ME': 월말, 'QE': 분기말)
            
        Returns:
            pd.Series: 일별 포트폴리오 수익률
        """
        portfolio_returns = pd.Series(index=returns.index, dtype=float)
        
        # 리밸런싱 날짜 계산
        rebalance_dates = pd.date_range(
            start=returns.index[0],
            end=returns.index[-1],
            freq=rebalancing_freq
        )
        
        # 실제 거래일과 매칭되는 리밸런싱 날짜 찾기
        actual_rebalance_dates = []
        for date in rebalance_dates:
            # 해당 월의 마지막 거래일 찾기
            month_end = returns.index[returns.index <= date][-1]
            if month_end not in actual_rebalance_dates:
                actual_rebalance_dates.append(month_end)
        
        # 각 리밸런싱 기간별로 수익률 계산
        for i in range(len(actual_rebalance_dates)-1):
            start_date = actual_rebalance_dates[i]
            end_date = actual_rebalance_dates[i+1]
            
            # 해당 기간의 가중치 찾기
            weight_date = weights.index[weights.index <= start_date][-1]
            current_weights = weights.loc[weight_date]
            
            # 해당 기간의 수익률 계산
            period_returns = returns.loc[start_date:end_date]
            period_returns = period_returns.iloc[1:]  # 시작일 수익률 제외
            
            # 일별 포트폴리오 수익률 계산
            daily_returns = (period_returns * current_weights).sum(axis=1)
            portfolio_returns.loc[period_returns.index] = daily_returns
        
        # 마지막 기간 처리
        if len(actual_rebalance_dates) > 0:
            last_date = actual_rebalance_dates[-1]
            weight_date = weights.index[weights.index <= last_date][-1]
            last_weights = weights.loc[weight_date]
            
            last_returns = returns.loc[last_date:]
            last_returns = last_returns.iloc[1:]  # 시작일 수익률 제외
            
            if not last_returns.empty:
                daily_returns = (last_returns * last_weights).sum(axis=1)
                portfolio_returns.loc[last_returns.index] = daily_returns
        
        return portfolio_returns.dropna()

    def calculate_portfolio_metrics(self, 
                                  returns: pd.Series,
                                  weights: Optional[pd.DataFrame] = None,
                                  benchmark_returns: Optional[pd.Series] = None,
                                  result_dir: str = None) -> Dict[str, float]:
        """
        포트폴리오 성과 지표를 계산하고 저장합니다.
        
        Args:
            returns (pd.Series): 일별 수익률
            weights (pd.DataFrame, optional): 포트폴리오 가중치
            benchmark_returns (pd.Series, optional): 벤치마크 수익률
            result_dir (str): 결과 저장 경로
            
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
        
        # 7. Win Rate (양의 수익률 비율)
        win_rate = (returns > 0).mean()
        
        # 8. Turnover (가중치 변화량)
        turnover = 0.0
        if weights is not None:
            weight_changes = weights.diff().abs().sum(axis=1)
            turnover = weight_changes.mean()
        
        # 9. Beta (시장과의 상관성)
        beta = np.nan
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            covariance = returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance != 0 else np.nan
        
        metrics = {
            'E(R)': mean_return,
            'Std(R)': std_return,
            'Sharpe Ratio': sharpe_ratio,
            'DD(R)': downside_dev,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown': max_drawdown,
            '% of +Ret': win_rate,
            'Turnover': turnover,
            'Beta': beta
        }
        
        # 결과 저장
        if result_dir is not None:
            metrics_df = pd.DataFrame([metrics])
            metrics_df.index = [returns.name if returns.name else 'Portfolio']
            
            # 저장 경로 생성
            metrics_dir = os.path.join(result_dir, 'metrics')
            os.makedirs(metrics_dir, exist_ok=True)
            
            # CSV 파일로 저장
            save_path = os.path.join(metrics_dir, f"{returns.name}_metrics.csv")
            metrics_df.to_csv(save_path)
            self.logger.info(f"Metrics saved to {save_path}")
            
            # LaTeX 형식으로도 저장
            latex_path = os.path.join(metrics_dir, f"{returns.name}_metrics.tex")
            metrics_df.style.format({
                'E(R)': '{:.2%}',
                'Std(R)': '{:.2%}',
                'Sharpe Ratio': '{:.2f}',
                'DD(R)': '{:.2%}',
                'Sortino Ratio': '{:.2f}',
                'Max Drawdown': '{:.2%}',
                '% of +Ret': '{:.2%}',
                'Turnover': '{:.2f}',
                'Beta': '{:.2f}'
            }).to_latex(latex_path)
            self.logger.info(f"LaTeX metrics saved to {latex_path}")
        
        return metrics

    def save_metrics_latex(self, 
                          metrics: pd.DataFrame,
                          result_dir: str,
                          model_name: str) -> None:
        """
        성과 지표를 LaTeX 형식으로 저장합니다.
        
        Args:
            metrics (pd.DataFrame): 성과 지표가 담긴 DataFrame
            result_dir (str): 결과 저장 경로
            model_name (str): 모델 이름
        """
        latex_path = os.path.join(result_dir, f'metrics_{model_name}.tex')
        
        # 지표별 포맷 설정 (모두 소수점 4자리)
        format_dict = {
            'E(R)': '{:.4%}',
            'Std(R)': '{:.4%}',
            'Sharpe Ratio': '{:.4f}',
            'DD(R)': '{:.4%}',
            'Sortino Ratio': '{:.4f}',
            'Max Drawdown': '{:.4%}',
            '% of +Ret': '{:.4%}',
            'Turnover': '{:.4f}',
            'Beta': '{:.4f}'
        }
        
        # Win Rate를 % of +Ret로 변경
        if 'Win Rate' in metrics.columns:
            metrics = metrics.rename(columns={'Win Rate': '% of +Ret'})
        
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
                    fmt = format_dict.get(col, '{:.4f}')  # 기본값도 4자리 소수점
                    row.append(fmt.format(value))
                f.write(" & ".join(row) + " \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}")
        
        # CSV 형식으로도 저장
        csv_path = os.path.join(result_dir, f'metrics_{model_name}.csv')
        metrics.to_csv(csv_path, float_format='%.4f')
        
        self.logger.info(f"Metrics saved to {latex_path} and {csv_path}")

    def calculate_turnover(self, 
                          weights: pd.DataFrame,
                          result_dir: str,
                          model_name: str) -> pd.Series:
        weight_changes = weights.diff().abs().sum(axis=1)
        turnover = weight_changes.fillna(0)
        
        # CSV 형식으로도 저장
        csv_path = os.path.join(result_dir, f'turnover_{model_name}.csv')
        turnover.to_csv(csv_path, float_format='%.4f')

        return turnover
