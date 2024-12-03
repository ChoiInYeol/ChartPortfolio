"""포트폴리오 시각화 모듈입니다."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import os
import scienceplots
import matplotlib.pyplot as plt
plt.style.use('science')

class PortfolioVisualizer:
    """포트폴리오 시각화를 위한 클래스입니다."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 기본 색상 스키마 
        self.color_scheme = {
            # 벤치마크
            'Naive': '#FF1F5B',  # qualitative 1
            'CNN Top': '#00CD6C',  # qualitative 2
            
            # 최적화
            'Max Sharpe': '#009ADE',  # qualitative 3 
            'Min Variance': '#AF58BA',  # qualitative 4
            'Min CVaR': '#FFC61E',  # qualitative 5
            
            # 최적화 + CNN Top
            'CNN Top + Max Sharpe': '#F28522',  # qualitative 6
            'CNN Top + Min Variance': '#A0B1BA',  # qualitative 7
            'CNN Top + Min CVaR': '#A6761D',  # qualitative 8
            
            # 시계열 모델
            'GRU': '#E9002D',  # qualitative 9
            'TCN': '#FFAA00',  # qualitative 10
            'TRANSFORMER': '#00B000',  # qualitative 11
            
            # 시계열 모델 + CNN Top
            'CNN + GRU': '#C40F5B',  # qualitative 12
            'CNN + TCN': '#FD8D3C',  # qualitative 13
            'CNN + TRANSFORMER': '#089099'  # qualitative 14
        }

    def plot_portfolio_comparison(self,
                                returns_dict: pd.DataFrame,
                                title: str,
                                result_dir: str,
                                selected_portfolios: list = None):
        """포트폴리오 성과를 비교 시각화합니다."""
        plt.figure(figsize=(15, 8), dpi=100)
        
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
                    linewidth=1)
            
            # 리밸런싱 날짜에 점 표시 (실제 데이터의 인덱스와 일치하는 날짜만)
            rebalance_dates = pd.date_range(start=cumulative_returns.index[0],
                                          end=cumulative_returns.index[-1],
                                          freq='ME')
            # 실제 데이터의 인덱스와 교집합만 사용
            valid_dates = rebalance_dates[rebalance_dates.isin(cumulative_returns.index)]
            
            if len(valid_dates) > 0:
                plt.plot(valid_dates,
                        cumulative_returns.loc[valid_dates, col],
                        'o',
                        color=color,
                        markersize=4)
        
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Returns', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.legend(bbox_to_anchor=(1.05, 1),
                  loc='upper left',
                  fontsize=10)
        
        plt.tight_layout()
        save_path = os.path.join(result_dir, 'figures', 'portfolio_comparison.png')
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
        plt.figure(figsize=(15, 8), dpi=100)
        
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
        save_path = os.path.join(os.path.dirname(save_path), 'figures', os.path.basename(save_path))
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Weight plot saved to {save_path}")
        
        