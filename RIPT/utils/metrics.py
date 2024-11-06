"""
포트폴리오 성과 측정을 위한 모듈입니다.
다양한 성과 지표 계산 기능을 포함합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

class PerformanceMetrics:
    """
    포트폴리오 성과 지표를 계산하는 클래스입니다.
    """
    
    def __init__(self):
        """
        PerformanceMetrics 초기화
        """
        self.logger = logging.getLogger(__name__)
        self.column_order = [
            'Naive', 'SPY', 'Top 100', 'Top N/10', 'Bottom 100',
            'Optimized_max_sharpe', 'Optimized_min_variance', 'Optimized_min_cvar'
        ]
        self.column_names = {
            'Naive': 'Naive',
            'SPY': 'SPY',
            'Top 100': 'Top 100',
            'Top N/10': 'Top N/10',
            'Bottom 100': 'Bottom 100',
            'Optimized_max_sharpe': 'Max sharpe',
            'Optimized_min_variance': 'Min Variance',
            'Optimized_min_cvar': 'Min CVaR'
        }

    def calculate_turnover(self, weights_dict: Dict) -> float:
        """
        포트폴리오 턴오버를 계산합니다.
        
        Args:
            weights_dict (Dict): 날짜별 포트폴리오 가중치
            
        Returns:
            float: 평균 턴오버 비율
        """
        dates = sorted(weights_dict.keys())
        turnover = 0
        
        for i in range(1, len(dates)):
            prev_weights = pd.Series(weights_dict[dates[i-1]])
            curr_weights = pd.Series(weights_dict[dates[i]])
            
            # 모든 종목을 포함하도록 인덱스 통합
            all_stocks = prev_weights.index.union(curr_weights.index)
            prev_weights = prev_weights.reindex(all_stocks, fill_value=0)
            curr_weights = curr_weights.reindex(all_stocks, fill_value=0)
            
            # 턴오버 계산 (단방향)
            turnover += np.abs(curr_weights - prev_weights).sum()
        
        return turnover / len(dates)

    def calculate_stock_turnover(self, stocks_dict: Dict) -> float:
        """
        주식 구성 변화율을 계산합니다.
        
        Args:
            stocks_dict (Dict): 날짜별 보유 주식 목록
            
        Returns:
            float: 평균 주식 턴오버 비율
        """
        dates = sorted(stocks_dict.keys())
        turnover = 0
        
        for i in range(1, len(dates)):
            prev_stocks = set(stocks_dict[dates[i-1]])
            curr_stocks = set(stocks_dict[dates[i]])
            
            changes = len(curr_stocks - prev_stocks) + len(prev_stocks - curr_stocks)
            turnover += changes / len(prev_stocks.union(curr_stocks))
        
        return turnover / (len(dates) - 1)

    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        최대 낙폭을 계산합니다.
        
        Args:
            returns (pd.Series): 수익률 시계열
            
        Returns:
            float: 최대 낙폭
        """
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()

    def calculate_metrics(self,
                         combined_returns: pd.DataFrame,
                         portfolio_weights: Dict,
                         selected_stocks: Dict,
                         n_stocks_10: int) -> pd.DataFrame:
        """
        모든 성과 지표를 계산합니다.
        
        Args:
            combined_returns (pd.DataFrame): 전체 포트폴리오 수익률
            portfolio_weights (Dict): 최적화된 포트폴리오 가중치
            selected_stocks (Dict): 선택된 주식 목록
            n_stocks_10 (int): Top N/10 주식 수
            
        Returns:
            pd.DataFrame: 계산된 성과 지표
        """
        metrics = {}
        
        for column in self.column_order:
            if column not in combined_returns.columns:
                continue
                
            returns = combined_returns[column].pct_change().dropna()
            
            metrics[self.column_names[column]] = {
                'Return': returns.mean() * 252,
                'Std': returns.std() * np.sqrt(252),
                'SR': (returns.mean() / returns.std()) * np.sqrt(252),
                'Max Drawdown': self.calculate_max_drawdown(returns)
            }
            
            # 턴오버 계산
            if column in ['Top 100', f'Top {n_stocks_10}', 'Bottom 100']:
                metrics[self.column_names[column]]['Turnover'] = self.calculate_stock_turnover(
                    selected_stocks[column]
                )
            elif column.startswith('Optimized_'):
                method = '_'.join(column.split('_')[1:])
                metrics[self.column_names[column]]['Turnover'] = self.calculate_turnover(
                    portfolio_weights[method]
                )
            else:
                metrics[self.column_names[column]]['Turnover'] = np.nan

        return pd.DataFrame(metrics).T[['Return', 'Std', 'SR', 'Max Drawdown', 'Turnover']]

    def save_metrics(self,
                    metrics: pd.DataFrame,
                    model: str,
                    window_size: int,
                    folder_name: str):
        """
        계산된 성과 지표를 다양한 형식으로 저장합니다.
        
        Args:
            metrics (pd.DataFrame): 성과 지표
            model (str): 모델 이름
            window_size (int): 윈도우 크기
            folder_name (str): 저장 폴더 경로
        """
        # CSV 저장
        csv_path = f'{folder_name}/performance_metrics_{model}{window_size}.csv'
        metrics.to_csv(csv_path, float_format='%.3f')
        
        # LaTeX 저장
        latex_path = f'{folder_name}/performance_metrics_{model}{window_size}.tex'
        with open(latex_path, 'w') as f:
            f.write("\\begin{table}[]\n\\begin{tabular}{ccccc}\n\\hline\n")
            f.write("             & Return   & Std      & SR       & Turnover \\\\ \\hline\n")
            for index, row in metrics.iterrows():
                f.write(f"{index:<12} & {row['Return']:.3f} & {row['Std']:.3f} & "
                       f"{row['SR']:.3f} & {row['Turnover']:.3f} \\\\\n")
            f.write("\\hline\n\\end{tabular}\n\\end{table}")
        
        # 텍스트 파일 저장
        txt_path = f'{folder_name}/performance_metrics_{model}{window_size}.txt'
        with open(txt_path, 'w') as f:
            f.write("\tReturn\tStd\tSR\tTurnover\n")
            for index, row in metrics.iterrows():
                f.write(f"{index}\t{row['Return']:.3f}\t{row['Std']:.3f}\t"
                       f"{row['SR']:.3f}\t{row['Turnover']:.3f}\n")
        
        self.logger.info(f"Performance metrics saved to {folder_name}")