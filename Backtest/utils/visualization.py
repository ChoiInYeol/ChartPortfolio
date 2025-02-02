"""포트폴리오 시각화 모듈입니다."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import os
import scienceplots
import matplotlib.pyplot as plt
from typing import Dict, List
from .metrics import PerformanceMetrics

plt.style.use('science')

class PortfolioVisualizer:
    """포트폴리오 시각화를 위한 클래스입니다."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 포트폴리오 그룹별 색상 스키마
        self.color_scheme = {
            # 벤치마크 (빨간색 계열)
            'Naive': '#FF0000',
            'CNN Top': '#FF6B6B',
            
            # 전통적 벤치마크 전략 (갈색 계열)
            'MOM': '#8B4513',  # 새들 브라운
            'STR': '#A0522D',  # 시에나
            'WSTR': '#CD853F', # 페루
            'TREND': '#DEB887', # 버블우드
            
            # 최적화 (파란색 계열)
            'Max Sharpe': '#0000FF',
            'Min Variance': '#4169E1',
            'Min CVaR': '#1E90FF',
            
            # CNN + 최적화 (초록색 계열)
            'CNN Top + Max Sharpe': '#008000',
            'CNN Top + Min Variance': '#32CD32',
            'CNN Top + Min CVaR': '#90EE90',
            
            # 시계열 모델 (보라색 계열)
            'GRU': '#800080',
            'TCN': '#BA55D3',
            'TRANSFORMER': '#DDA0DD',
            
            # CNN + 시계열 모델 (주황색 계열)
            'CNN + GRU': '#FFA500',
            'CNN + TCN': '#FFB84D',
            'CNN + TRANSFORMER': '#FFD700'
        }
        
        # 포트폴리오 표시 순서
        self.portfolio_order = [
            # 벤치마크
            'Naive',
            'CNN Top',
            # 전통적 벤치마크 전략
            'MOM',
            'STR',
            'WSTR',
            'TREND',
            # 최적화
            'Max Sharpe',
            'Min Variance',
            'Min CVaR',
            # CNN + 최적화
            'CNN Top + Max Sharpe',
            'CNN Top + Min Variance',
            'CNN Top + Min CVaR',
            # 시계열 모델
            'GRU',
            'TCN',
            'TRANSFORMER',
            # CNN + 시계열 모델
            'CNN + GRU',
            'CNN + TCN',
            'CNN + TRANSFORMER'
        ]

    def plot_portfolio_comparison(self,
                                returns_dict: pd.DataFrame,
                                title: str = "Portfolio Performance Comparison",
                                result_dir: str = None,
                                selected_portfolios: List[str] = None):
        """Portfolio performance visualization."""
        plt.figure(figsize=(15, 8), dpi=300)
        
        # 누적 수익률 계산
        cumulative_returns = (1 + returns_dict).cumprod()
        
        # 포트폴리오 순서대로 플롯
        plotted_portfolios = []
        for portfolio_name in selected_portfolios:
            if portfolio_name in cumulative_returns.columns:
                plotted_portfolios.append(portfolio_name)
                color = self.color_scheme.get(portfolio_name, '#333333')
                plt.plot(cumulative_returns.index, 
                        cumulative_returns[portfolio_name], 
                        label=portfolio_name,
                        color=color,
                        linewidth=1,
                        alpha=0.8)
        
        # 그래프 스타일링
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Returns', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 범례 추가
        if plotted_portfolios:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        # 저장
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
        
        # 저장 경로 확인 및 생성
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Weight plot saved to {save_path}")
        
    def plot_strategy_comparison(self,
                               metrics_df: pd.DataFrame,
                               result_dir: str,
                               title: str = "Strategy Performance Comparison"):
        """
        단일 전략과 복합 전략의 성과를 비교하는 bar plot을 생성하고 저장합니다.
        
        Args:
            metrics_df (pd.DataFrame): 성과 지표가 담긴 DataFrame
            result_dir (str): 결과 저장 경로
            title (str): 그래프 제목
        """
        # 전략 그룹 정의
        single_stage = ['Max Sharpe', 'Min Variance', 'Min CVaR', 'GRU', 'TCN', 'TRANSFORMER']
        two_stage = ['CNN Top + Max Sharpe', 'CNN Top + Min Variance', 'CNN Top + Min CVaR',
                     'CNN + GRU', 'CNN + TCN', 'CNN + TRANSFORMER']
        
        # 실제 존재하는 포트폴리오만 필터링
        available_single = [p for p in single_stage if p in metrics_df.index]
        available_two = [p for p in two_stage if p in metrics_df.index]
        
        if not available_single and not available_two:
            self.logger.warning("No portfolios available for comparison")
            return
        
        # 비교할 지표 선택
        metrics = ['E(R)', 'Std(R)', 'Sharpe Ratio', 'Max Drawdown']
        
        # 그래프 생성
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # 각 그룹의 데이터 준비
            single_data = metrics_df.loc[available_single, metric] if available_single else pd.Series()
            two_stage_data = metrics_df.loc[available_two, metric] if available_two else pd.Series()
            
            # Max Drawdown의 경우 양수로 변환하여 표시
            if metric == 'Max Drawdown':
                if not single_data.empty:
                    single_data = single_data.abs()
                if not two_stage_data.empty:
                    two_stage_data = two_stage_data.abs()
            
            # Bar 위치 설정
            width = 0.35
            max_bars = max(len(single_data), len(two_stage_data))
            x = np.arange(max_bars)
            
            # Single Stage 바 그리기
            if not single_data.empty:
                ax.bar(x[:len(single_data)] - width/2, single_data.values, 
                      width, label='Single Stage', color='lightblue', alpha=0.8)
                # 값 표시
                for i, v in enumerate(single_data):
                    value = v if metric != 'Max Drawdown' else -v
                    ax.text(i - width/2, v, f'{value:.2%}' if metric != 'Sharpe Ratio' else f'{value:.2f}',
                           ha='center', va='bottom')
            
            # Two Stage 바 그리기
            if not two_stage_data.empty:
                ax.bar(x[:len(two_stage_data)] + width/2, two_stage_data.values, 
                      width, label='Two Stage', color='lightcoral', alpha=0.8)
                # 값 표시
                for i, v in enumerate(two_stage_data):
                    value = v if metric != 'Max Drawdown' else -v
                    ax.text(i + width/2, v, f'{value:.2%}' if metric != 'Sharpe Ratio' else f'{value:.2f}',
                           ha='center', va='bottom')
            
            # 축 설정
            ax.set_title(metric, fontsize=12, pad=10)
            ax.set_xticks(x)
            
            # x축 레이블 설정
            if not single_data.empty:
                labels = single_data.index
            elif not two_stage_data.empty:
                labels = two_stage_data.index
            else:
                labels = []
            
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
        
        # 전체 타이틀 추가
        fig.suptitle(title, fontsize=14, y=1.02)
        
        # 범례 추가 (존재하는 그룹만)
        legend_labels = []
        if available_single:
            legend_labels.append('Single Stage')
        if available_two:
            legend_labels.append('Two Stage')
        
        if legend_labels:
            fig.legend(legend_labels, 
                      loc='upper right', 
                      bbox_to_anchor=(0.99, 1.01),
                      ncol=len(legend_labels))
        
        plt.tight_layout()
        
        # 저장 경로 설정
        figures_dir = os.path.join(result_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        save_path = os.path.join(figures_dir, 'strategy_comparison.png')
        
        # 그래프 저장
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        self.logger.info(f"Strategy comparison plot saved to {save_path}")
        
        # 지표 데이터도 CSV로 저장
        metrics_dir = os.path.join(result_dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
        metrics_path = os.path.join(metrics_dir, 'strategy_comparison_metrics.csv')
        metrics_df.to_csv(metrics_path, float_format='%.4f')
        self.logger.info(f"Strategy comparison metrics saved to {metrics_path}")
        
        # LaTeX 형식으로도 저장
        latex_path = os.path.join(metrics_dir, 'strategy_comparison_metrics.tex')
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
        self.logger.info(f"Strategy comparison LaTeX metrics saved to {latex_path}")
        
    def plot_top_returns_weights_comparison(self,
                                          returns: pd.DataFrame,
                                          portfolio_weights: Dict[str, pd.DataFrame],
                                          investment_start: str,
                                          forward_period: int = 20,
                                          top_n: int = 100,
                                          save_path: str = None,
                                          title: str = None):
        """
        모든 포트폴리오의 상위 수익률 종목 비중을 비교하는 bar plot을 생성합니다.
        """
        plt.figure(figsize=(15, 8), dpi=300)
        
        # 각 포트폴리오의 상위 종목 평균 비중 계산
        mean_weights = {}
        
        # returns의 거래일 인덱스
        trading_days = returns[returns.index >= investment_start].index
        
        # 전체 기간 누적 수익률 계산 (미래 수익률)
        forward_returns = pd.DataFrame(index=trading_days, columns=returns.columns, dtype=float)
        
        for date in trading_days[:-forward_period]:  # 마지막 forward_period 일은 제외
            try:
                # 해당 시점부터 forward_period 동의 수익률 계산
                end_idx = returns.index.get_loc(date) + forward_period
                if end_idx >= len(returns.index):
                    continue
                end_date = returns.index[end_idx]
                period_returns = returns.loc[date:end_date]
                
                # 누적 수익률 계산 (데이터 타입을 float으로 강제 변환)
                cum_returns = (1 + period_returns.astype(float)).prod() - 1
                forward_returns.loc[date] = cum_returns
            except Exception as e:
                self.logger.warning(f"Error calculating returns at {date}: {str(e)}")
                continue
        
        # 상위 수익률 종목 선택 (전체 기간)
        total_returns = forward_returns.mean().astype(float)
        top_stocks = total_returns.nlargest(top_n).index
        
        # 각 포트폴리오의 상위 종목 비중 계산
        for name, weights in portfolio_weights.items():
            try:
                # 거래일에 맞춰 가중치 리샘플링
                oos_weights = weights.reindex(trading_days, method='ffill')
                
                # 상위 수익률 종목들의 비중 합계
                top_weights = oos_weights[top_stocks].astype(float).sum(axis=1)
                mean_weights[name] = top_weights.mean()
            except Exception as e:
                self.logger.warning(f"Error calculating weights for {name}: {str(e)}")
                continue
        
        if not mean_weights:
            self.logger.warning("No valid weights calculated")
            return
        
        # 평균 비중 기준으로 내림차순 정렬
        mean_weights = pd.Series(mean_weights).sort_values(ascending=False)
        
        # Bar plot 생성 (portfolio_comparison의 색상 스키마 사용)
        bars = plt.bar(range(len(mean_weights)), mean_weights.values,
                      color=[self.color_scheme.get(name, '#333333') for name in mean_weights.index])
        
        # 축 설정
        plt.xticks(range(len(mean_weights)), mean_weights.index, rotation=45, ha='right')
        plt.ylabel('Average Weight in Top Return Stocks')
        plt.title(title or f'Average Weight in Top {top_n} Return Stocks by Portfolio')
        
        # 값 표시
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 저장
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            self.logger.info(f"Top returns weights comparison plot saved to {save_path}")
        
        