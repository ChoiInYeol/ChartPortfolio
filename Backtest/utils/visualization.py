"""포트폴리오 시각화 모듈입니다."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import os
import scienceplots
plt.style.use(['science'])

class PortfolioVisualizer:
    """포트폴리오 시각화를 위한 클래스입니다."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 기본 색상 스키마
        self.color_scheme = {
            'Naive': '#1f77b4',
            'CNN Top': '#ff7f0e',
            'Max Sharpe': '#d62728',
            'Min Variance': '#9467bd',
            'Min CVaR': '#8c564b',
            'CNN Top + Max Sharpe': '#e377c2',
            'CNN Top + Min Variance': '#7f7f7f',
            'CNN Top + Min CVaR': '#bcbd22',
            'GRU': '#17becf',
            'TCN': '#1a55FF',
            'TRANSFORMER': '#FF55AA',
            'CNN + GRU': '#AA55FF',
            'CNN + TCN': '#55FFAA',
            'CNN + TRANSFORMER': '#FFAA55'
        }

    def plot_portfolio_comparison(self,
                                returns_dict: pd.DataFrame,
                                title: str,
                                result_dir: str,
                                selected_portfolios: list = None):
        """
        포트폴리오 성과를 비교 시각화합니다.
        
        Args:
            returns_dict (pd.DataFrame): 포트폴리오별 수익률
            title (str): 그래프 제목
            result_dir (str): 결과 저장 경로
            selected_portfolios (list, optional): 표시할 포트폴리오 목록
        """
        plt.figure(figsize=(15, 8), dpi=300)
        
        # 선택된 포트폴리오만 필터링
        if selected_portfolios is not None:
            returns_dict = returns_dict[selected_portfolios]
        
        # 누적 수익률 계산
        cumulative_returns = pd.DataFrame(index=returns_dict.index)
        for col in returns_dict.columns:
            cumulative_returns[col] = (1 + returns_dict[col]).cumprod()
        
        # 각 포트폴리오 플롯
        for col in cumulative_returns.columns:
            color = self.color_scheme.get(col, '#333333')
            plt.plot(cumulative_returns.index, 
                    cumulative_returns[col], 
                    label=col,
                    color=color,
                    linewidth=1.5)
        
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Returns', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.legend(bbox_to_anchor=(1.05, 1),
                  loc='upper left',
                  fontsize=10)
        
        plt.tight_layout()
        save_path = os.path.join(result_dir, 'portfolio_comparison.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Portfolio comparison plot saved to {save_path}")

    def plot_weights(self,
                    weights: pd.DataFrame,
                    save_path: str,
                    title: str = "Portfolio Weights"):
        """
        포트폴리오 가중치를 플롯합니다.
        
        Args:
            weights (pd.DataFrame): 포트폴리오 가중치
            save_path (str): 저장 경로
            title (str): 그래프 제목
        """
        plt.figure(figsize=(15, 8), dpi=300)
        
        # 0이 아닌 가중치만 선택
        weights = weights.loc[:, (weights != 0).any()]
        
        # 월별 리샘플링
        weights_monthly = weights.resample('ME').last()
        
        # HSV 색공간에서 고르게 분포된 색상 생성
        n_colors = len(weights.columns)
        colors = [plt.cm.hsv(i/n_colors) for i in range(n_colors)]
        
        # 리밸런싱 날짜 계산 및 세로선 그리기
        rebalance_dates = pd.date_range(
            start=weights.index[0],
            end=weights.index[-1],
            freq='ME'
        )
        
        # 리밸런싱 세로선 그리기
        for date in rebalance_dates:
            if date in weights_monthly.index:
                plt.axvline(x=date, color='gray', linestyle='--', alpha=0.3)
        
        # Bar plot 생성
        bar_width = 20
        bottom = np.zeros(len(weights_monthly))
        
        for i, col in enumerate(weights_monthly.columns):
            plt.bar(weights_monthly.index,
                   weights_monthly[col],
                   bottom=bottom,
                   width=bar_width,
                   color=colors[i],
                   alpha=0.8)
            bottom += weights_monthly[col]
        
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Weight', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Weight plot saved to {save_path}")
        
        