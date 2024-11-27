"""포트폴리오 성과 시각화를 위한 모듈입니다."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import os
import scienceplots
plt.style.use(['science'])

class PortfolioVisualizer:
    """포트폴리오 성과 시각화를 위한 클래스입니다."""
    
    def __init__(self):
        """PortfolioVisualizer 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 기본 색상 스키마
        self.color_scheme = {
            'Naive': '#1f77b4',
            'CNN Top 100': '#ff7f0e',
            'CNN Top 50 + Max Sharpe': '#2ca02c',
            'CNN Top 50 + Min Variance': '#d62728',
            'CNN Top 50 + Min CVaR': '#9467bd',
            'GRU Top 50': '#8c564b',
            'TCN Top 50': '#e377c2',
            'TRANSFORMER Top 50': '#7f7f7f'
        }

    def plot_portfolio_comparison(self,
                                returns_dict: pd.DataFrame,
                                title: str,
                                result_dir: str) -> None:
        """
        포트폴리오 성과를 비교 시각화합니다.
        
        Args:
            returns_dict (pd.DataFrame): 포트폴리오별 수익률
            title (str): 그래프 제목
            result_dir (str): 결과 저장 경로
        """
        plt.figure(figsize=(12, 6), dpi=300)
        
        # 각 포트폴리오 플롯
        for col in returns_dict.columns:
            color = self.color_scheme.get(col, '#333333')
            plt.plot(returns_dict.index, returns_dict[col],
                    label=col,
                    color=color,
                    linewidth=1.5)
        
        # 그래프 스타일링
        plt.title(title, fontsize=12)
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Cumulative Returns', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        save_path = os.path.join(result_dir, 'portfolio_comparison.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Portfolio comparison plot saved to {save_path}")
        
        