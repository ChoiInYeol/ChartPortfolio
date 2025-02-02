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
        
        # 가중치와 수익률의 유효 기간 확인
        valid_start = max(returns.index[0], weights.index[0])
        valid_end = min(returns.index[-1], weights.index[-1])
        
        # 리밸런싱 날짜 계산
        rebalance_dates = pd.date_range(
            start=valid_start,
            end=valid_end,
            freq=rebalancing_freq
        )
        
        # 실제 거래일과 매칭되는 리밸런싱 날짜 찾기
        actual_rebalance_dates = []
        for date in rebalance_dates:
            # 해당 월의 마지막 거래일 찾기
            month_dates = returns.index[returns.index <= date]
            if len(month_dates) > 0:
                month_end = month_dates[-1]
                if month_end not in actual_rebalance_dates:
                    actual_rebalance_dates.append(month_end)
        
        if not actual_rebalance_dates:
            self.logger.warning("No valid rebalancing dates found")
            return portfolio_returns.round(4)
        
        # 각 리밸런싱 기간별로 수익률 계산
        for i in range(len(actual_rebalance_dates)-1):
            start_date = actual_rebalance_dates[i]
            end_date = actual_rebalance_dates[i+1]
            
            # 해당 기간의 가중치 찾기
            available_dates = weights.index[weights.index <= start_date]
            if len(available_dates) == 0:
                continue
            
            weight_date = available_dates[-1]
            current_weights = weights.loc[weight_date]
            
            # 해당 기간의 수익률 계산
            period_returns = returns.loc[start_date:end_date]
            period_returns = period_returns.iloc[1:]  # 시작일 수익률 제외
            
            if period_returns.empty:
                continue
            
            # 일별 포트폴리오 수익률 계산
            daily_returns = (period_returns * current_weights).sum(axis=1)
            portfolio_returns.loc[period_returns.index] = daily_returns
        
        # 마지막 기간 처리
        if len(actual_rebalance_dates) > 0:
            last_date = actual_rebalance_dates[-1]
            available_dates = weights.index[weights.index <= last_date]
            
            if len(available_dates) > 0:
                weight_date = available_dates[-1]
                last_weights = weights.loc[weight_date]
                
                last_returns = returns.loc[last_date:]
                last_returns = last_returns.iloc[1:]  # 시작일 수익률 제외
                
                if not last_returns.empty:
                    daily_returns = (last_returns * last_weights).sum(axis=1)
                    portfolio_returns.loc[last_returns.index] = daily_returns
        
        return portfolio_returns.dropna().round(4)

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
        mean_return = round(returns.mean() * annual_factor, 4)
        
        # 2. Standard Deviation (연율화)
        std_return = round(returns.std() * np.sqrt(annual_factor), 4)
        
        # 3. Sharpe Ratio
        sharpe_ratio = round((mean_return - risk_free_rate) / std_return if std_return != 0 else 0, 4)
        
        # 4. Downside Deviation
        downside_returns = returns[returns < 0]
        downside_dev = round(downside_returns.std() * np.sqrt(annual_factor) if len(downside_returns) > 0 else 0, 4)
        
        # 5. Sortino Ratio
        sortino_ratio = round((mean_return - risk_free_rate) / downside_dev if downside_dev != 0 else 0, 4)
        
        # 6. Maximum Drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_drawdown = round(drawdowns.min(), 4)
        
        # 7. Win Rate (양의 수익률 비율)
        win_rate = round((returns > 0).mean(), 4)
        
        # 8. Turnover (가중치 변화량)
        turnover = 0.0
        if weights is not None:
            # 월별 리밸런싱 가중치 추출
            monthly_weights = weights.resample('ME').last()
            
            # 리밸런싱 시점의 가중치 변화 계산
            # 매수/매도 모두 포함하기 위해 변화량의 절대값 합계를 2로 나눔
            weight_changes = monthly_weights.diff().abs().sum(axis=1) / 2
            
            # 연간 turnover 계산 (월별 평균 × 12)
            turnover = round(weight_changes.mean() * 12, 4)
        
        # 9. Beta (시장과의 상관성)
        beta = np.nan
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            covariance = returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            beta = round(covariance / benchmark_variance if benchmark_variance != 0 else np.nan, 4)
        
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
            metrics_df.to_csv(save_path, float_format='%.4f')
            self.logger.info(f"Metrics saved to {save_path}")
            
            # LaTeX 형식으로도 저장
            latex_path = os.path.join(metrics_dir, f"{returns.name}_metrics.tex")
            metrics_df.style.format({
                'E(R)': '{:.4%}',
                'Std(R)': '{:.4%}',
                'Sharpe Ratio': '{:.4f}',
                'DD(R)': '{:.4%}',
                'Sortino Ratio': '{:.4f}',
                'Max Drawdown': '{:.4%}',
                '% of +Ret': '{:.4%}',
                'Turnover': '{:.4f}',
                'Beta': '{:.4f}'
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
        
        # 포트폴리오 이름 LaTeX 포맷 매핑
        name_mapping = {
            'Naive': 'Naive',
            'CNN Top': 'CNN Top',
            'MOM': 'MOM (2-12)',
            'STR': 'STR (1M)',
            'WSTR': 'WSTR (1W)',
            'TREND': 'TREND',
            'Max Sharpe': 'Max Sharpe',
            'Min Variance': 'Min Variance',
            'Min CVaR': 'Min CVaR',
            'CNN Top + Max Sharpe': 'CNN+Max Sharpe',
            'CNN Top + Min Variance': 'CNN+Min Var',
            'CNN Top + Min CVaR': 'CNN+Min CVaR',
            'GRU': 'GRU',
            'TCN': 'TCN',
            'TRANSFORMER': 'Transformer',
            'CNN + GRU': 'CNN+GRU',
            'CNN + TCN': 'CNN+TCN',
            'CNN + TRANSFORMER': 'CNN+Transformer'
        }
        
        # 지표별 포맷 설정 (모두 소수점 4자리)
        format_dict = {
            'E(R)': lambda x: f"{x:.4%}".rstrip('0').rstrip('.') + '%',
            'Std(R)': lambda x: f"{x:.4%}".rstrip('0').rstrip('.') + '%',
            'Sharpe Ratio': lambda x: f"{x:.4f}".rstrip('0').rstrip('.'),
            'DD(R)': lambda x: f"{x:.4%}".rstrip('0').rstrip('.') + '%',
            'Sortino Ratio': lambda x: f"{x:.4f}".rstrip('0').rstrip('.'),
            'Max Drawdown': lambda x: f"{x:.4%}".rstrip('0').rstrip('.') + '%',
            '% of +Ret': lambda x: f"{x:.4%}".rstrip('0').rstrip('.') + '%',
            'Turnover': lambda x: f"{x:.4f}".rstrip('0').rstrip('.'),
            'Beta': lambda x: f"{x:.4f}".rstrip('0').rstrip('.')
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
            
            # 포트폴리오 이름을 LaTeX 포맷으로 변환
            portfolio_names = [name_mapping.get(idx, idx) for idx in metrics.index]
            f.write("Portfolio & " + " & ".join(portfolio_names) + " \\\\\n")
            f.write("\\hline\n")
            
            for col in metrics.columns:
                row = [col]
                for idx in metrics.index:
                    value = metrics.loc[idx, col]
                    fmt = format_dict.get(col, lambda x: f"{x:.4f}".rstrip('0').rstrip('.'))
                    row.append(fmt(value))
                f.write(" & ".join(row) + " \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}")
        
        # CSV 형식으로도 저장
        csv_path = os.path.join(result_dir, f'metrics_{model_name}.csv')
        metrics.to_csv(csv_path, float_format='%.4g')  # 일반화된 포맷 사용
        
        self.logger.info(f"Metrics saved to {latex_path} and {csv_path}")

    def calculate_turnover(self, 
                          weights: pd.DataFrame,
                          result_dir: str,
                          model_name: str) -> pd.Series:
        weight_changes = weights.diff().abs().sum(axis=1)
        turnover = weight_changes.fillna(0).round(4)
        
        # CSV 형식으로도 저장
        csv_path = os.path.join(result_dir, f'turnover_{model_name}.csv')
        turnover.to_csv(csv_path, float_format='%.4f')

        return turnover
