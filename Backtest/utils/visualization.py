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
        
        # 모델 이름 매핑 추가
        self.model_names = {
            'Naive': 'Naive',
            'top100_equal': 'CNN Top 100',
            'bottom100_equal': 'CNN Bottom 100',
            'max_sharpe': 'CNN Top 50 + Max Sharpe'
        }
        
        # 모델별 색상 동적 할당
        model_colors = {
            'GRU': plt.cm.Set2(0),
            'TCN': plt.cm.Set2(1),
            'TRANSFORMER': plt.cm.Set2(2)
        }
        
        # 기본 색상 스키마 (벤치마크)
        self.color_scheme = {
            'Naive': plt.cm.Greys(0.8),
            'CNN Top 100': plt.cm.Greys(0.6),
            'CNN Bottom 100': plt.cm.Greys(0.4),
            'CNN Top 50 + Max Sharpe': plt.cm.Greys(0.2)
        }
        
        # 모델 색상 자동 생성 함수
        def generate_model_colors(model_name: str, n_select: int) -> Dict[str, np.ndarray]:
            base_color = model_colors[model_name]
            cnn_color = plt.cm.Set2(len(model_colors) + list(model_colors.keys()).index(model_name))
            return {
                f"{model_name} Top {n_select}": base_color,
                f"CNN + {model_name} Top {n_select}": cnn_color
            }
        
        # 모델별 색상 스키마 확장
        for model in ['GRU', 'TCN', 'TRANSFORMER']:
            for n_select in [20, 30, 50, 100]:  # 가능한 n_select 값들
                self.color_scheme.update(generate_model_colors(model, n_select))
                # 모델 이름 매핑도 추가
                self.model_names.update({
                    f"{model} Top {n_select}": f"{model} Top {n_select}",
                    f"CNN + {model} Top {n_select}": f"CNN + {model} Top {n_select}"
                })
        
        # 투명도 설정
        self.alpha_settings = {
            'main': 0.9,  # 기본 선
            'net': 0.4,   # Net 수익률
            'marker': 0.6  # 리밸런싱 마커
        }
        
        # 선 스타일 설정
        self.line_styles = {
            'benchmark': '--',  # 벤치마크용 점선
            'model': '-'       # 모델용 실선
        }
        
        # 선 두께 설정
        self.line_widths = {
            'benchmark': 1.5,  # 벤치마크용 얇은 선
            'model': 2.0      # 모델용 두꺼운 선
        }

    def setup_plot(self, figsize: tuple = (8, 6), dpi: int = 400) -> None:
        """
        플롯 기본 설정을 기화합니다.
        
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
                             weights_dict: Dict[str, pd.DataFrame],
                             result_dir: str,
                             include_net: bool = False) -> None:
        """모든 포트폴리오의 턴오버를 한 그래프에 시각화합니다."""
        plt.figure(figsize=(16, 8))
        
        # Net returns 포함 여부에 따라 포트폴리오 필터링
        if include_net:
            # Net returns만 선택
            plot_weights = {name: weights for name, weights in weights_dict.items() 
                           if '(Net)' in name}
            suffix = '_with_net'
        else:
            # Gross returns만 선택
            plot_weights = {name: weights for name, weights in weights_dict.items() 
                           if '(Net)' not in name}
            suffix = ''
        
        for model_name, weights in plot_weights.items():
            # 월별 턴오버 계산
            monthly_weights = weights.resample('ME').last()
            turnovers = []
            dates = []
            
            for i in range(1, len(monthly_weights)):
                prev_weights = monthly_weights.iloc[i-1]
                curr_weights = monthly_weights.iloc[i]
                
                # 모든 종목을 포함하도록 인덱스 통합
                all_stocks = prev_weights.index.union(curr_weights.index)
                prev_weights = prev_weights.reindex(all_stocks, fill_value=0)
                curr_weights = curr_weights.reindex(all_stocks, fill_value=0)
                
                turnover = np.abs(curr_weights - prev_weights).sum() / 2
                turnovers.append(turnover)
                dates.append(monthly_weights.index[i])
            
            display_name = self.model_names.get(model_name.split(' (Net)')[0], model_name)
            color = self.color_scheme.get(display_name, '#333333')
            
            plt.plot(dates, turnovers,
                    label=model_name,
                    color=color,
                    alpha=self.alpha_settings['main'])
        
        plt.title('Monthly Portfolio Turnover Comparison')
        plt.xlabel('Date')
        plt.ylabel('Turnover Ratio')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f'turnover_comparison{suffix}.png'), bbox_inches='tight')
        plt.close()

    def plot_rolling_metrics(self,
                            returns: pd.DataFrame,
                            window: int = 252,
                            result_dir: str = './Backtest/',
                            include_net: bool = False) -> None:
        """롤링 성과 지표를 시각화합니다."""
        if include_net:
            # Net returns만 선택
            plot_data = returns[[col for col in returns.columns if '(Net)' in col]]
            suffix = '_with_net'
        else:
            # Gross returns만 선택
            plot_data = returns[[col for col in returns.columns if '(Net)' not in col]]
            suffix = ''
        
        plt.figure(figsize=(16, 8))
        
        # 수익률 기반으로 롤링 샤프 비율 계산
        for name, ret in plot_data.items():
            display_name = self.model_names.get(name.split(' (Net)')[0], name)
            if '(Net)' not in name:  # Net 수익률 제외
                # 일간 수익률로 변환
                daily_returns = ret.pct_change(fill_method='pad').dropna()
                
                # 롤링 샤프 비율 계산
                rolling_mean = daily_returns.rolling(window=window).mean() * 252
                rolling_std = daily_returns.rolling(window=window).std() * np.sqrt(252)
                rolling_sharpe = (rolling_mean - 0.02) / rolling_std  # 무위험 수익률 2% 가정
                
                color = self.color_scheme.get(display_name, '#333333')
                plt.plot(rolling_sharpe.index, rolling_sharpe.values,
                        label=display_name,
                        color=color,
                        alpha=self.alpha_settings['main'])
        
        plt.title(f'Rolling Sharpe Ratio ({window} days)')
        plt.xlabel('Date')
        plt.ylabel('Sharpe Ratio')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', ncol=2)
        plt.tight_layout()
        
        save_path = os.path.join(result_dir, f'rolling_sharpe{suffix}.png')
        plt.savefig(save_path)
        plt.close()

    def plot_drawdown_analysis(self,
                              returns: pd.DataFrame,
                              result_dir: str,
                              include_net: bool = False) -> None:
        """낙폭 분석을 시각화합니다."""
        if include_net:
            # Net returns만 선택
            plot_data = returns[[col for col in returns.columns if '(Net)' in col]]
            suffix = '_with_net'
        else:
            # Gross returns만 선택
            plot_data = returns[[col for col in returns.columns if '(Net)' not in col]]
            suffix = ''
        
        self.setup_plot(figsize=(15, 5))
        
        for column in plot_data.columns:
            drawdown = (plot_data[column] - plot_data[column].cummax()) / plot_data[column].cummax()
            plt.plot(drawdown.index, drawdown, label=column)
        
        plt.title('Drawdown Analysis')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(result_dir, f'drawdown_analysis{suffix}.png')
        plt.savefig(save_path)
        plt.close()

    def plot_weight_evolution(self,
                             weights: pd.DataFrame,
                             result_dir: str,
                             model_name: str,
                             plot_type: str = 'area',
                             resample_freq: str = 'ME') -> None:
        """포트폴리오 비중의 시간에 따른 변화를 시각화합니다."""
        # 가중치가 0이 아닌 종목만 선택
        weights = weights.loc[:, (weights != 0).any()]
        weights_resampled = weights.resample(resample_freq).last()
        
        plt.figure(figsize=(16, 8), dpi=300)
        
        # 리밸런싱 날짜 계산 및 세로선 그리기
        rebalance_dates = pd.date_range(
            start=weights.index[0],
            end=weights.index[-1],
            freq=resample_freq
        )
        
        # 리밸런싱 세로선 그리기
        for date in rebalance_dates:
            if date in weights_resampled.index:
                plt.axvline(x=date, color='gray', linestyle='--', alpha=0.3)
        
        # HSV 색공간에서 고르게 분포된 색상 생성
        n_colors = len(weights.columns)
        colors = [plt.cm.hsv(i/n_colors) for i in range(n_colors)]
        
        if plot_type == 'area':
            # 스택 플롯 생성 (모든 종목)
            plt.stackplot(weights_resampled.index,
                         weights_resampled.T,
                         colors=colors,
                         alpha=0.8)
            
            # 각 리밸런싱 시점의 비중 합계 출력
            for date in rebalance_dates:
                if date in weights_resampled.index:
                    total_weight = weights_resampled.loc[date].sum()
                    plt.text(date, 1.02, f'{total_weight:.2f}',
                            horizontalalignment='center',
                            verticalalignment='bottom',
                            fontsize=8,
                            rotation=45)
        
        else:  # bar plot
            bar_width = 20
            bottom = np.zeros(len(weights_resampled))
            
            for i, col in enumerate(weights_resampled.columns):
                plt.bar(weights_resampled.index,
                       weights_resampled[col],
                       bottom=bottom,
                       width=bar_width,
                       color=colors[i],
                       alpha=0.8)
                bottom += weights_resampled[col]
        
        display_name = self.model_names.get(model_name, model_name)
        plt.title(f'Portfolio Weight Evolution - {display_name}')
        plt.xlabel('Date')
        plt.ylabel('Weight')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # 파일명에 net 포함 여부 반영
        save_path = os.path.join(result_dir, f'weight_evolution_{model_name}_{plot_type}.png')
        
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Weight evolution plot saved for {model_name} ({plot_type})")

    def plot_confusion_matrix(self,
                             metrics_df: pd.DataFrame,
                             result_dir: str,
                             model_name: str) -> None:
        """
        혼동 행렬 시각화를 생성합니다.
        
        Args:
            metrics_df (pd.DataFrame): 혼동 행렬 지표
            result_dir (str): 결과 저장 경로
            model_name (str): 모델 이름
        """
        # 전체 합계 계산
        total_matrix = {
            'TP': metrics_df['TP'].sum(),
            'FP': metrics_df['FP'].sum(),
            'TN': metrics_df['TN'].sum(),
            'FN': metrics_df['FN'].sum()
        }
        
        # 혼동 행렬 생성
        cm = np.array([[total_matrix['TN'], total_matrix['FP']],
                       [total_matrix['FN'], total_matrix['TP']]])
        
        plt.figure(figsize=(8, 6), dpi=300)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {model_name}')
        
        # 레이블 설정
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        classes = ['Down', 'Up']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # 값 표시
        thresh = cm.max() / 2
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, f'confusion_matrix_{model_name}.png'))
        plt.close()

    def plot_portfolio_comparison(self,
                                  returns_dict: pd.DataFrame,
                                  title: str,
                                  result_dir: str,
                                  include_net: bool = False) -> None:
        """포트폴리오 성과를 비교 시각화합니다."""
        if include_net:
            # Net returns만 선택
            plot_data = returns_dict[[col for col in returns_dict.columns if '(Net)' in col]]
            suffix = '_with_net'
        else:
            # Gross returns만 선택
            plot_data = returns_dict[[col for col in returns_dict.columns if '(Net)' not in col]]
            suffix = ''
        
        plt.figure(figsize=(16, 8), dpi=300)
        
        # 데이터 검증
        for col in plot_data.columns:
            if plot_data[col].isnull().all():
                self.logger.warning(f"Portfolio {col} contains all NaN values")
                plot_data = plot_data.drop(columns=[col])
        
        # 벤치마크 포트폴리오 순서 정의
        benchmark_names = ['Naive', 'CNN Top 100', 'CNN Bottom 100', 'CNN Top 50 + Max Sharpe']
        if include_net:
            benchmark_names.extend([name + ' (Net)' for name in benchmark_names])
        
        model_names = [name for name in plot_data.columns 
                      if name not in benchmark_names and 
                      (include_net or '(Net)' not in name)]
        
        # 색상 설정
        color_dict = {}
        
        # 벤치마크 색상 (회색 계열)
        for i, name in enumerate(benchmark_names):
            if name in plot_data.columns:
                color_dict[name] = plt.cm.Greys(0.3 + i * 0.15)  # 더 뚜렷한 색상 차이
        
        # 모델 색상 (컬러풀한 색상)
        colors = plt.cm.tab20(np.linspace(0, 1, len(model_names)))
        for i, name in enumerate(model_names):
            color_dict[name] = colors[i]
        
        # 벤치마크 플롯
        for name in benchmark_names:
            if name in plot_data.columns:
                plt.plot(plot_data.index, plot_data[name],
                        label=name,
                        color=color_dict[name],
                        linestyle='--',
                        linewidth=1.5,
                        alpha=0.8)
        
        # 모델 플롯
        for name in model_names:
            plt.plot(plot_data.index, plot_data[name],
                    label=name,
                    color=color_dict[name],
                    linestyle='-',
                    linewidth=2,
                    alpha=0.9)
        
        # 리밸런싱 마커
        rebalance_dates = pd.date_range(
            start=plot_data.index[0],
            end=plot_data.index[-1],
            freq='ME'
        )
        for date in rebalance_dates:
            if date in plot_data.index:
                plt.plot(date, plot_data.loc[date, name],
                        marker='o',
                        color=color_dict[name],
                        markersize=4,
                        alpha=0.6)
        
        # 그래프 스타일링
        plt.title(title, fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Returns', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # x축 레이블 조정
        plt.xticks(rotation=45)
        
        # 범례 설정 (그래프 내부 우측 상단)
        plt.legend(loc='upper left',
                  bbox_to_anchor=(0.02, 0.98),
                  fontsize=10,
                  ncol=2)
        
        plt.tight_layout()
        save_path = os.path.join(result_dir, f'portfolio_comparison{suffix}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Portfolio comparison plot saved to {save_path}")