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
                                weights_dict: Dict[str, pd.DataFrame] = None,
                                up_prob: pd.DataFrame = None,
                                title: str = "Portfolio Performance Comparison",
                                result_dir: str = None,
                                investment_start: str = None,
                                rebalancing_freq: str = 'ME',
                                include_costs: bool = True,
                                commission_rate: float = 0.0003,
                                selected_portfolios: List[str] = None):
        """Portfolio performance visualization."""
        plt.figure(figsize=(15, 8), dpi=300)
        
        # investment_start가 None인 경우 첫 번째 날짜 사용
        if investment_start is None:
            investment_start = returns_dict.index[0]
        else:
            investment_start = pd.to_datetime(investment_start)
        
        # Out of Sample 기간 데이터만 사용
        returns_dict = returns_dict[returns_dict.index >= investment_start]
        
        # 리밸런싱 날짜 계산
        period_end_dates = pd.date_range(
            start=investment_start,
            end=returns_dict.index[-1],
            freq=rebalancing_freq
        )
        
        # 리밸런싱 주기에 따른 라벨 설정
        if rebalancing_freq == 'WE':
            period_label = "Weekly"
        elif rebalancing_freq == 'ME':
            period_label = "Monthly"
        elif rebalancing_freq == 'QE':
            period_label = "Quarterly"
        elif rebalancing_freq == '2QE':
            period_label = "Semi-Annual"
        else:  # 'YE'
            period_label = "Annual"
        
        # 매수/매도 날짜 계산
        buy_dates = []
        sell_dates = []
        
        for period_end in period_end_dates:
            if rebalancing_freq == 'WE':
                period_start = period_end - pd.offsets.Week(1)
            elif rebalancing_freq == 'ME':
                period_start = period_end - pd.offsets.MonthBegin(1)
            elif rebalancing_freq == 'QE':
                period_start = period_end - pd.offsets.QuarterBegin(1)
            elif rebalancing_freq == '2QE':
                period_start = period_end - pd.offsets.QuarterBegin(2)
            else:  # 'YE'
                period_start = period_end - pd.offsets.YearBegin(1)
            
            # 거래일 찾기
            if up_prob is not None:
                try:
                    buy_date = up_prob.index[up_prob.index >= period_start][0]
                    sell_date = up_prob.index[up_prob.index <= period_end][-1]
                except IndexError:
                    continue
            else:
                # up_prob가 없는 경우 returns_dict의 인덱스 사용
                try:
                    buy_date = returns_dict.index[returns_dict.index >= period_start][0]
                    sell_date = returns_dict.index[returns_dict.index <= period_end][-1]
                except IndexError:
                    continue
            
            buy_dates.append(buy_date)
            sell_dates.append(sell_date)
        
        # 선택된 포트폴리오만 필터링
        if selected_portfolios is not None:
            available_portfolios = [p for p in selected_portfolios if p in returns_dict.columns]
            if not available_portfolios:
                self.logger.warning("No selected portfolios found in returns data")
                return
            returns_dict = returns_dict[available_portfolios]
        
        # 거래비용 반영
        if include_costs and weights_dict is not None:
            metrics = PerformanceMetrics()
            returns_with_costs = {}
            
            for portfolio_name in returns_dict.columns:
                if portfolio_name in weights_dict:
                    returns_with_costs[portfolio_name] = metrics.calculate_portfolio_returns(
                        returns=returns_dict[[portfolio_name]],
                        weights=weights_dict[portfolio_name],
                        rebalancing_freq=rebalancing_freq,
                        transaction_cost=commission_rate
                    )
            
            returns_dict = pd.DataFrame(returns_with_costs)
        
        # 누적 수익률 계산
        cumulative_returns = (1 + returns_dict).cumprod()
        
        # 포트폴리오 순서대로 플롯
        plotted_portfolios = []
        for portfolio_name in self.portfolio_order:
            if portfolio_name in cumulative_returns.columns:
                plotted_portfolios.append(portfolio_name)
                color = self.color_scheme.get(portfolio_name, '#333333')
                plt.plot(cumulative_returns.index, 
                        cumulative_returns[portfolio_name], 
                        label=f"{portfolio_name}{' (with costs)' if include_costs else ''}",
                        color=color,
                        linewidth=1,
                        alpha=0.8)
        
        # 리밸런싱 날짜 표시
        if buy_dates and sell_dates:
            for date in buy_dates:
                plt.axvline(x=date, color='blue', linestyle='--', alpha=0.1, 
                           label=f'Buy ({period_label})' if date == buy_dates[0] else "")
            for date in sell_dates:
                plt.axvline(x=date, color='red', linestyle='--', alpha=0.1, 
                           label=f'Sell ({period_label})' if date == sell_dates[0] else "")
        
        # 그래프 스타일링
        plt.title(f"{title}{' (Including Transaction Costs)' if include_costs else ''}", 
                  fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Returns', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 범례 추가
        if plotted_portfolios:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        # 저장
        suffix = '_with_costs' if include_costs else ''
        save_path = os.path.join(result_dir, 'figures', f'portfolio_comparison{suffix}.png')
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
        
        # 저장 경로 확인 및 생성
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Weight plot saved to {save_path}")
        
    def plot_top_weights(self,
                        weights: pd.DataFrame,
                        save_path: str,
                        title: str = "Portfolio Top Weights",
                        top_n: int = 30):
        """
        상위 N개 종목의 포트폴리오 가중치를 플롯합니다.
        
        Args:
            weights (pd.DataFrame): 포트폴리오 가중치
            save_path (str): 저장 경로
            title (str): 그래프 제목
            top_n (int): 표시할 상위 종목 수
        """
        with plt.style.context(['science', 'no-latex']):
            plt.figure(figsize=(15, 8), dpi=100)
            
            # 월별 리샘플링
            weights_monthly = weights.resample('ME').last()
            
            # 상위 N개 종목 선택
            mean_weights = weights.mean()
            top_tickers = mean_weights.nlargest(top_n).index
            
            # 색상 팔레트 설정 (상위 종목 + Others)
            colors = plt.cm.tab20(np.linspace(0, 1, top_n + 1))
            color_mapping = {ticker: colors[i] for i, ticker in enumerate(top_tickers)}
            color_mapping['Others'] = colors[-1]
            
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
            
            # 상위 종목 먼저 플롯
            for ticker in top_tickers:
                if (weights_monthly[ticker] != 0).any():
                    plt.bar(weights_monthly.index,
                        weights_monthly[ticker],
                        bottom=bottom,
                        width=bar_width,
                        color=color_mapping[ticker],
                        label=rf"{ticker} [{mean_weights[ticker]:.1%}]",  # raw string으로 변경
                        alpha=0.8)
                    bottom += weights_monthly[ticker]
            
            # 나머지 종목들을 'Others'로 통합
            other_tickers = [col for col in weights_monthly.columns if col not in top_tickers]
            if other_tickers:
                others = weights_monthly[other_tickers].sum(axis=1)
                if (others != 0).any():
                    others_mean = mean_weights[other_tickers].sum()
                    plt.bar(weights_monthly.index,
                        others,
                        bottom=bottom,
                        width=bar_width,
                        color=color_mapping['Others'],
                        label=rf"Others [{others_mean:.1%}]",  # raw string으로 변경
                        alpha=0.8)
            
            plt.title(title, fontsize=14, pad=20)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Weight', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            # 범례 추가 (2열로 표시)
            plt.legend(bbox_to_anchor=(1.05, 1),
                    loc='upper left',
                    fontsize=8,  # 폰트 크기를 줄임
                    ncol=2)
            
            plt.tight_layout()
            # _topn 접미사 추가
            base_path = os.path.splitext(save_path)[0]
            new_save_path = f"{base_path}_top{top_n}.png"
            plt.savefig(new_save_path, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Top {top_n} weight plot saved to {new_save_path}")
        
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
        
        # 비교할 지표 선택
        metrics = ['E(R)', 'Std(R)', 'Sharpe Ratio', 'DD(R)']
        
        # 그래프 생성
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=100)
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # 각 그룹의 데이터 준비
            single_data = metrics_df.loc[single_stage, metric]
            two_stage_data = metrics_df.loc[two_stage, metric]
            
            # Bar 위치 설정
            x = np.arange(max(len(single_data), len(two_stage_data)))
            width = 0.35
            
            # Bar plot 생성
            ax.bar(x - width/2, single_data, width, label='Single Stage', color='lightblue', alpha=0.8)
            ax.bar(x + width/2, two_stage_data, width, label='Two Stage', color='lightcoral', alpha=0.8)
            
            # 축 설정
            ax.set_title(metric, fontsize=12, pad=10)
            ax.set_xticks(x)
            ax.set_xticklabels(single_data.index, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # 값 표시
            for i, v in enumerate(single_data):
                ax.text(i - width/2, v, f'{v:.2%}' if metric != 'Sharpe Ratio' else f'{v:.2f}',
                       ha='center', va='bottom')
            for i, v in enumerate(two_stage_data):
                ax.text(i + width/2, v, f'{v:.2%}' if metric != 'Sharpe Ratio' else f'{v:.2f}',
                       ha='center', va='bottom')
        
        # 전체 타이틀 추가
        fig.suptitle(title, fontsize=14, y=1.02)
        
        # 범례 추가
        fig.legend(['Single Stage', 'Two Stage'], 
                  loc='upper right', 
                  bbox_to_anchor=(0.99, 1.01),
                  ncol=2)
        
        plt.tight_layout()
        
        # 저장 경로 설정
        figures_dir = os.path.join(result_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        save_path = os.path.join(figures_dir, 'strategy_comparison.png')
        
        # 그래프 저장
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close()
        
        self.logger.info(f"Strategy comparison plot saved to {save_path}")
        
        # 지표 데이터도 CSV로 저장
        metrics_dir = os.path.join(result_dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
        metrics_path = os.path.join(metrics_dir, 'strategy_comparison_metrics.csv')
        metrics_df.to_csv(metrics_path)
        self.logger.info(f"Strategy comparison metrics saved to {metrics_path}")
        
        # LaTeX 형식으로도 저장
        latex_path = os.path.join(metrics_dir, 'strategy_comparison_metrics.tex')
        metrics_df.style.format({
            'Annual Return': '{:.2%}',
            'Annual Volatility': '{:.2%}',
            'Sharpe Ratio': '{:.2f}',
            'Max Drawdown': '{:.2%}'
        }).to_latex(latex_path)
        self.logger.info(f"Strategy comparison LaTeX metrics saved to {latex_path}")
        
    def plot_top_returns_weights(self,
                               returns: pd.DataFrame,
                               weights: pd.DataFrame,
                               investment_start: str,
                               lookback_period: int = 60,
                               top_n: int = 100,
                               save_path: str = None,
                               title: str = None):
        """
        각 시점별 상위 수익률 종목들의 포트폴리오 비중을 시각화합니다.
        """
        plt.figure(figsize=(15, 8), dpi=100)
        
        # 거래일 인덱��� 사용
        trading_days = returns[returns.index >= investment_start].index
        
        # 가중치를 거래일에 맞춰 리샘플링
        oos_weights = weights.reindex(trading_days, method='ffill')
        
        # 각 시점별 상위 종목들의 가중치 합계 계산
        top_weights_sum = pd.Series(index=trading_days, dtype=float)
        
        for date in trading_days:
            try:
                # 해당 시점까지의 과거 수익률 계산
                end_date = date
                start_idx = returns.index.get_loc(date) - lookback_period
                if start_idx < 0:  # lookback_period가 데이터 시작일보다 이전인 경우
                    continue
                start_date = returns.index[start_idx]
                period_returns = returns.loc[start_date:end_date]
                
                # 해당 기간 누적 수익률 계산
                cum_returns = (1 + period_returns).prod() - 1
                
                # 상위 수익률 종목 선택
                top_stocks = cum_returns.nlargest(top_n).index
                
                # 해당 시점의 상위 종목 비중 합계
                if date in oos_weights.index:  # 날짜 존재 여부 확인
                    top_weights_sum[date] = oos_weights.loc[date, top_stocks].sum()
            except Exception as e:
                self.logger.warning(f"Error calculating weights at {date}: {str(e)}")
                continue
        
        # 결측치 제거
        top_weights_sum = top_weights_sum.dropna()
        
        if len(top_weights_sum) == 0:
            self.logger.warning("No valid data points to plot")
            return
        
        # 그래프 그리기
        plt.plot(top_weights_sum.index, top_weights_sum, 
                label=f'Rolling Top {top_n} Stocks Weight',
                color='blue', linewidth=2)
        
        # 평균 비중 표시
        mean_weight = top_weights_sum.mean()
        plt.axhline(y=mean_weight, color='r', linestyle='--', 
                   label=f'Average Weight ({mean_weight:.2%})')
        
        # 스타일링
        plt.title(title or f'Portfolio Weight in Rolling Top {top_n} Return Stocks', 
                 fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Portfolio Weight', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.legend()
        
        # 저장
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Rolling top returns weights plot saved to {save_path}")
        
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
        plt.figure(figsize=(15, 8), dpi=100)
        
        # 각 포트폴리오의 상위 종목 평균 비중 계산
        mean_weights = {}
        
        # returns의 거래일 인덱스
        trading_days = returns[returns.index >= investment_start].index
        
        # 전체 기간 누적 수익률 계산 (미래 수익률)
        forward_returns = pd.DataFrame(index=trading_days, columns=returns.columns, dtype=float)
        
        for date in trading_days[:-forward_period]:  # 마지막 forward_period 일은 제외
            try:
                # 해당 시점부터 forward_period 동안의 수익률 계산
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
        
        # 포트폴리오 그룹별 색상 설정
        colors = []
        for portfolio in mean_weights.index:
            color = self.color_scheme.get(portfolio, '#333333')
            colors.append(color)
        
        # Bar plot 생성
        bars = plt.bar(range(len(mean_weights)), mean_weights.values, color=colors)
        
        # 축 설정
        plt.xticks(range(len(mean_weights)), mean_weights.index, rotation=45, ha='right')
        plt.ylabel('Average Weight in Rolling Top Return Stocks')
        plt.title(title or f'Average Weight in Rolling Top {top_n} Return Stocks by Portfolio')
        
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
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Rolling top returns weights comparison plot saved to {save_path}")
        
        