"""
포트폴리오 성과 시각화를 위한 모듈입니다.
다양한 시각화 기능을 포함합니다.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
import os
import scienceplots
plt.style.use(['science'])

class PortfolioVisualizer:
    """
    포트폴리오 성과 시각화를 위한 클래스입니다.
    """
    
    def __init__(self):
        """
        PortfolioVisualizer 초기화
        """
        self.logger = logging.getLogger(__name__)
        plt.style.use(['science'])
        
    def setup_plot(self, figsize: tuple = (8, 6), dpi: int = 400) -> None:
        """
        플롯 기본 설정을 초기화합니다.
        
        Args:
            figsize (tuple): 그래프 크기
            dpi (int): 해상도
        """
        plt.figure(figsize=figsize, dpi=dpi)

    def plot_optimized_portfolios(self,
                                portfolio_ret: pd.DataFrame,
                                optimized_portfolios: Dict[str, pd.Series],
                                model: str,
                                window_size: int,
                                result_dir: str,
                                rebalance_dates: Dict[str, List[pd.Timestamp]]) -> None:
        """
        최적화된 포트폴리오의 성과를 시각화합니다.
        
        Args:
            portfolio_ret (pd.DataFrame): 기본 포트폴리오 수익률
            optimized_portfolios (Dict[str, pd.Series]): 최적화된 포트폴리오 수익률
            model (str): 모델 이름
            window_size (int): 윈도우 크기
            result_dir (str): 결과 저장 경로
            rebalance_dates (Dict[str, List[pd.Timestamp]]): 리밸런싱 날짜
        """
        self.setup_plot()
        
        # 색상 설정
        colors = plt.cm.tab10(np.linspace(0, 1, len(portfolio_ret.columns) + len(optimized_portfolios)))
        
        # 기본 포트폴리오 플롯
        self._plot_base_portfolios(portfolio_ret, colors, rebalance_dates)
        
        # 최적화된 포트폴리오 플롯
        self._plot_optimized_portfolios(optimized_portfolios, colors, len(portfolio_ret.columns), rebalance_dates)
        
        # 그래프 스타일링
        self._style_plot(model, window_size)
        
        # 저장
        self._save_plot(result_dir, model, window_size)

    def _plot_base_portfolios(self,
                            portfolio_ret: pd.DataFrame,
                            colors: np.ndarray,
                            rebalance_dates: Dict[str, List[pd.Timestamp]]) -> None:
        """
        기본 포트폴리오를 플롯합니다.
        """
        for i, column in enumerate(portfolio_ret.columns):
            plt.plot(portfolio_ret.index, portfolio_ret[column], 
                    label=column, color=colors[i])
            
            # 리밸런싱 날짜 표시
            if column in rebalance_dates:
                for date in rebalance_dates[column]:
                    if date in portfolio_ret.index:
                        plt.plot(date, portfolio_ret.loc[date, column], 
                               marker='o', color=colors[i], markersize=3)

    def _plot_optimized_portfolios(self,
                                 optimized_portfolios: Dict[str, pd.Series],
                                 colors: np.ndarray,
                                 color_offset: int,
                                 rebalance_dates: Dict[str, List[pd.Timestamp]]) -> None:
        """
        최적화된 포트폴리오를 플롯합니다.
        """
        for i, (method, returns) in enumerate(optimized_portfolios.items(), start=color_offset):
            cumulative_returns = (1 + returns).cumprod()
            plt.plot(cumulative_returns.index, cumulative_returns.values, 
                    label=f'Optimized ({method})', color=colors[i], linestyle='--')
            
            # 리밸런싱 날짜 표시
            for date in rebalance_dates['Top 100']:
                if date in cumulative_returns.index:
                    plt.plot(date, cumulative_returns.loc[date], 
                           marker='o', color=colors[i], markersize=3)

    def _style_plot(self, model: str, window_size: int) -> None:
        """
        그래프 스타일을 설정합니다.
        """
        plt.title(f'Portfolio Performance Comparison - {model} {window_size}-day', 
                 fontsize=12)
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Cumulative Returns', fontsize=10)
        plt.legend(fontsize=8, ncol=2, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    def _save_plot(self, result_dir: str, model: str, window_size: int) -> None:
        """
        그래프를 저장합니다.
        """
        save_path = os.path.join(result_dir, f'portfolio_comparison_{model}{window_size}.png')
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Plot saved to {save_path}")

    def plot_weight_distribution(self,
                               portfolio_weights: Dict[pd.Timestamp, Dict[str, float]],
                               model: str,
                               window_size: int,
                               result_dir: str) -> None:
        """
        포트폴리오 가중치 분포를 시각화합니다.
        
        Args:
            portfolio_weights (Dict): 포트폴리오 가중치
            model (str): 모델 이름
            window_size (int): 윈도우 크기
            result_dir (str): 결과 저장 경로
        """
        self.setup_plot()
        
        dates = sorted(portfolio_weights.keys())
        weights_array = np.array([list(portfolio_weights[date].values()) for date in dates])
        
        plt.boxplot(weights_array.T, whis=1.5)
        plt.title(f'Portfolio Weight Distribution - {model} {window_size}-day', 
                 fontsize=12)
        plt.xlabel('Time Period', fontsize=10)
        plt.ylabel('Weight', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(result_dir, f'weight_distribution_{model}{window_size}.png')
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Weight distribution plot saved to {save_path}")

    def plot_turnover_analysis(self,
                             portfolio_weights: Dict[pd.Timestamp, Dict[str, float]],
                             model: str,
                             window_size: int,
                             result_dir: str) -> None:
        """
        포트폴리오 턴오버를 시각화합니다.
        
        Args:
            portfolio_weights (Dict): 포트폴리오 가중치
            model (str): 모델 이름
            window_size (int): 윈도우 크기
            result_dir (str): 결과 저장 경로
        """
        self.setup_plot()
        
        dates = sorted(portfolio_weights.keys())[1:]
        turnovers = []
        
        for i in range(1, len(dates)):
            prev_weights = pd.Series(portfolio_weights[dates[i-1]])
            curr_weights = pd.Series(portfolio_weights[dates[i]])
            
            all_stocks = prev_weights.index.union(curr_weights.index)
            prev_weights = prev_weights.reindex(all_stocks, fill_value=0)
            curr_weights = curr_weights.reindex(all_stocks, fill_value=0)
            
            turnover = np.abs(curr_weights - prev_weights).sum()
            turnovers.append(turnover)
        
        plt.plot(dates[1:], turnovers, marker='o', markersize=3)
        plt.title(f'Portfolio Turnover Analysis - {model} {window_size}-day', 
                 fontsize=12)
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Turnover', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(result_dir, f'turnover_analysis_{model}{window_size}.png')
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Turnover analysis plot saved to {save_path}")