"""
포트폴리오 성과 측정을 위한 모듈입니다.
다양한 성과 지표 계산 기능을 포함합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
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

    def calculate_additional_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        추가적인 성과 지표들을 계산합니다.
        
        Args:
            returns (pd.Series): 수익률 시계열
            
        Returns:
            Dict[str, float]: 계산된 성과 지표들
        """
        # 월별 수익률로 변환
        monthly_returns = returns.resample('M').last().pct_change()
        
        # 승률 계산
        win_rate = (monthly_returns > 0).mean()
        
        # 최대 연속 손실 기간
        drawdown = (returns - returns.cummax()) / returns.cummax()
        max_drawdown_duration = self._calculate_max_drawdown_duration(drawdown)
        
        # 수익/위험 비율
        profit_risk_ratio = returns.mean() / returns.std()
        
        # 월별 양의 수익률 평균 / 음의 수익률 평균
        gain_loss_ratio = abs(monthly_returns[monthly_returns > 0].mean() / 
                             monthly_returns[monthly_returns < 0].mean())
        
        return {
            'Win Rate': win_rate,
            'Max Drawdown Duration': max_drawdown_duration,
            'Profit/Risk Ratio': profit_risk_ratio,
            'Gain/Loss Ratio': gain_loss_ratio
        }

    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """
        최대 연속 손실 기간을 계산합니다.
        """
        drawdown_periods = (drawdown < 0).astype(int)
        max_duration = 0
        current_duration = 0
        
        for is_drawdown in drawdown_periods:
            if is_drawdown:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
                
        return max_duration

    def calculate_portfolio_metrics(self, returns: pd.Series, weights: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        포트폴리오 성과 지표를 계산합니다.
        
        Args:
            returns (pd.Series): 수익��� 시계열
            weights (pd.DataFrame, optional): 포트폴리오 가중치
            
        Returns:
            Dict[str, float]: 계산된 성과 지표
        """
        # 연율화 팩터
        annual_factor = 252
        
        # 기본 통계량
        mean_return = returns.mean() * annual_factor
        std_return = returns.std() * np.sqrt(annual_factor)
        
        # 샤프 비율
        risk_free_rate = 0.02  # 연 2% 가정
        sharpe_ratio = (mean_return - risk_free_rate) / std_return
        
        # 소티노 비율
        downside_returns = returns[returns < 0]
        sortino_ratio = (mean_return - risk_free_rate) / (downside_returns.std() * np.sqrt(annual_factor))
        
        # 최대 낙폭
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # 칼마 비율
        calmar_ratio = mean_return / abs(max_drawdown)
        
        # 승률
        win_rate = (returns > 0).mean()
        
        # 수익/손실 비율
        gains = returns[returns > 0].mean()
        losses = returns[returns < 0].mean()
        profit_loss_ratio = abs(gains / losses) if losses != 0 else np.inf
        
        # 턴오버 계산 (가중치가 제공된 경우)
        turnover = 0.0
        if weights is not None:
            weight_changes = weights.diff().abs().sum(axis=1)
            turnover = weight_changes.mean()  # 평균 턴오버
        
        return {
            'Annual Return': mean_return,
            'Annual Volatility': std_return,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Profit/Loss Ratio': profit_loss_ratio,
            'Turnover': turnover
        }

    def save_metrics_latex(self, 
                          metrics: pd.DataFrame,
                          result_dir: str,
                          model_name: str) -> None:
        """성과 지표를 LaTeX 형식으로 저장합니다."""
        latex_path = os.path.join(result_dir, f'metrics_{model_name}.tex')
        
        # 각 지표별 최적값 찾기
        better_higher = ['Annual Return', 'Sharpe Ratio', 'Sortino Ratio', 
                        'Calmar Ratio', 'Win Rate', 'Profit/Loss Ratio']
        better_lower = ['Annual Volatility', 'Max Drawdown', 'Turnover']
        
        best_values = {}
        for col in metrics.columns:
            if col in better_higher:
                best_values[col] = metrics[col].max()
            elif col in better_lower:
                best_values[col] = metrics[col].min()
        
        with open(latex_path, 'w') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Portfolio Performance Metrics}\n")
            f.write("\\begin{tabular}{l" + "r" * len(metrics.columns) + "}\n")
            f.write("\\hline\n")
            
            # 헤더
            f.write("Metric & " + " & ".join(metrics.columns) + " \\\\\n")
            f.write("\\hline\n")
            
            # 각 지표별 행
            for metric in metrics.index:
                row = [metric]
                for col in metrics.columns:
                    value = metrics.loc[metric, col]
                    
                    # 값 포맷팅
                    if metric in better_higher:
                        if value == best_values[metric]:
                            cell = f"\\textcolor{{green}}{{\\textbf{{{value:.4f}}}}}"
                        elif value > metrics[col].mean():
                            cell = f"\\textbf{{{value:.4f}}}"
                        else:
                            cell = f"{value:.4f}"
                    elif metric in better_lower:
                        if value == best_values[metric]:
                            cell = f"\\textcolor{{green}}{{\\underline{{{value:.4f}}}}}"
                        elif value < metrics[col].mean():
                            cell = f"\\underline{{{value:.4f}}}"
                        else:
                            cell = f"{value:.4f}"
                    else:
                        cell = f"{value:.4f}"
                    
                    row.append(cell)
                
                f.write(" & ".join(row) + " \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}")
        
        # CSV 형식으로도 저장
        csv_path = os.path.join(result_dir, f'metrics_{model_name}.csv')
        metrics.to_csv(csv_path, float_format='%.4f')

    def calculate_prediction_metrics(self,
                                   up_prob_df: pd.DataFrame,
                                   returns_df: pd.DataFrame,
                                   prediction_window: int = 20,
                                   threshold: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        예측 성과를 계산하고 혼동 행렬을 생성합니다.
        
        Args:
            up_prob_df (pd.DataFrame): 상승확률 데이터프레임
            returns_df (pd.DataFrame): 수익률 데이터프레임
            prediction_window (int): 예측 기간 (일)
            threshold (float): 상승 예측을 위한 임계값
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - 날짜별 혼동 행렬 지표
                - 전체 성과 지표
        """
        try:
            # 1. 날짜 인덱스 정렬
            common_dates = up_prob_df.index.intersection(returns_df.index)
            up_prob_df = up_prob_df.loc[common_dates]
            returns_df = returns_df.loc[common_dates]
            
            # 2. 예측 결과와 실제 수익률 매핑
            metrics_df = pd.DataFrame(index=common_dates)
            
            for date in common_dates:
                # prediction_window 이후의 날짜가 없으면 건너뛰기
                future_idx = returns_df.index.get_loc(date) + prediction_window
                if future_idx >= len(returns_df.index):
                    continue
                    
                future_date = returns_df.index[future_idx]
                
                # 예측값과 실제값 계산
                predictions = up_prob_df.loc[date] > threshold
                actuals = returns_df.loc[date:future_date].prod() > 1  # 누적 수익률이 양수인지
                
                # 혼동 행렬 계산
                metrics_df.loc[date, 'TP'] = np.sum((predictions == True) & (actuals == True))
                metrics_df.loc[date, 'FP'] = np.sum((predictions == True) & (actuals == False))
                metrics_df.loc[date, 'TN'] = np.sum((predictions == False) & (actuals == False))
                metrics_df.loc[date, 'FN'] = np.sum((predictions == False) & (actuals == True))
            
            # 3. 전체 성과 지표 계산
            total_metrics = pd.Series({
                'Accuracy': (metrics_df['TP'].sum() + metrics_df['TN'].sum()) / 
                           (metrics_df['TP'].sum() + metrics_df['TN'].sum() + 
                            metrics_df['FP'].sum() + metrics_df['FN'].sum()),
                'Precision': metrics_df['TP'].sum() / (metrics_df['TP'].sum() + metrics_df['FP'].sum()),
                'Recall': metrics_df['TP'].sum() / (metrics_df['TP'].sum() + metrics_df['FN'].sum()),
                'F1': 2 * (metrics_df['TP'].sum() * metrics_df['TP'].sum()) / 
                      (2 * metrics_df['TP'].sum() + metrics_df['FP'].sum() + metrics_df['FN'].sum())
            })
            
            return metrics_df, total_metrics
            
        except Exception as e:
            self.logger.error(f'Error calculating prediction metrics: {str(e)}')
            return pd.DataFrame(), pd.Series()