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

    def calculate_portfolio_returns(self, returns: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
        """
        포트폴리오 수익률을 계산합니다.
        
        Args:
            returns (pd.DataFrame): 일별 수익률
            weights (pd.DataFrame): 포트폴리오 가중치
            
        Returns:
            pd.Series: 포트폴리오 수익률
        """
        try:
            # 날짜 인덱스 정렬
            aligned_weights = weights.reindex(index=returns.index, method='ffill')
            aligned_weights = aligned_weights.fillna(0)
            
            # 수익률 계산
            portfolio_returns = (returns * aligned_weights).sum(axis=1)
            return portfolio_returns.dropna()
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio returns: {str(e)}")
            return pd.Series(dtype=float)

    def calculate_portfolio_metrics(self, portfolio_returns: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        각 포트폴리오의 성과 지표를 계산합니다.
        
        Args:
            portfolio_returns (Dict[str, pd.Series]): 포트폴리오별 수익률
            
        Returns:
            pd.DataFrame: 포트폴리오별 성과 지표
        """
        try:
            metrics_list = []
            
            # 각 포트폴리오별로 지표 계산
            for name, returns in portfolio_returns.items():
                metrics = self._calculate_metrics(returns)
                metrics['Portfolio'] = name
                metrics_list.append(metrics)
            
            # 결과를 DataFrame으로 변환
            metrics_df = pd.DataFrame(metrics_list)
            
            # 필요한 지표만 선택
            selected_metrics = [
                'Portfolio',
                'E(R)',       # Annual Return
                'Std(R)',     # Annual Volatility
                'Sharpe Ratio',
                'DD(R)',      # Max Drawdown
                'Sortino Ratio',
                '% of +Ret',  # Hit Ratio
                'Turnover',
                'Beta'
            ]
            
            # 컬럼 이름 매핑
            metrics_df = metrics_df.rename(columns={
                'Annual Return': 'E(R)',
                'Annual Volatility': 'Std(R)',
                'Max Drawdown': 'DD(R)',
                'Hit Ratio': '% of +Ret'
            })
            
            # 포트폴리오 순서 정의
            portfolio_order = [
                # 벤치마크
                'CNN Top',
                # 팩터 타이밍
                'Factor Timing ND',
                'Factor Timing FM',
                'Factor Timing 1M',
                'Factor Timing 1MOpt',
                # 전통적 전략
                'MOM',
                'STR',
                'WSTR',
                'TREND',
                # 최적화
                'Max Sharpe',
                'Min Variance',
                'Min CVaR',
                'Target 6%',
                'Target 8%',
                'Target 10%',
                'Target 12%',
                # CNN + 최적화
                'CNN Top + Max Sharpe',
                'CNN Top + Min Variance',
                'CNN Top + Min CVaR',
                'CNN Top + Target 6%',
                'CNN Top + Target 8%',
                'CNN Top + Target 10%',
                'CNN Top + Target 12%'
            ]
            
            # 존재하는 포트폴리오만 필터링
            available_portfolios = [p for p in portfolio_order if p in metrics_df['Portfolio'].values]
            
            # 포트폴리오 순서대로 정렬
            metrics_df['Portfolio'] = pd.Categorical(
                metrics_df['Portfolio'],
                categories=available_portfolios,
                ordered=True
            )
            metrics_df = metrics_df.sort_values('Portfolio')
            
            # 필요한 지표만 선택하고 인덱스 설정
            metrics_df = metrics_df[selected_metrics].set_index('Portfolio')
            
            return metrics_df
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return pd.DataFrame()

    def _calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        개별 포트폴리오의 성과 지표를 계산합니다.
        
        Args:
            returns (pd.Series): 일별 수익률
            
        Returns:
            Dict[str, float]: 성과 지표
        """
        try:
            # 연율화 계수
            annual_factor = np.sqrt(252)  # 일별 데이터 기준
            
            # 1. 기본 통계량
            mean_return = returns.mean() * 252  # 연간 수익률
            std_return = returns.std() * annual_factor  # 연간 변동성
            
            # 2. 샤프 비율
            risk_free_rate = 0.02  # 연간 2%로 가정
            daily_rf = (1 + risk_free_rate) ** (1/252) - 1
            excess_returns = returns - daily_rf
            sharpe_ratio = (mean_return - risk_free_rate) / std_return if std_return != 0 else 0
            
            # 3. 최대 낙폭
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # 4. 승률
            win_rate = len(returns[returns > 0]) / len(returns)
            
            # 5. 소티노 비율
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * annual_factor if len(downside_returns) > 0 else 0
            sortino_ratio = (mean_return - risk_free_rate) / downside_std if downside_std != 0 else 0
            
            # 6. 베타 (시장 수익률 대비)
            market_returns = returns  # 여기서는 간단히 동일 수익률 사용
            covariance = returns.cov(market_returns)
            market_variance = market_returns.var()
            beta = covariance / market_variance if market_variance != 0 else 1
            
            # 7. 턴오버 (여기서는 0으로 설정, 실제로는 가중치 변화로 계산)
            turnover = 0.0
            
            metrics = {
                'Annual Return': mean_return,
                'Annual Volatility': std_return,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown': max_drawdown,
                'Sortino Ratio': sortino_ratio,
                'Hit Ratio': win_rate,
                'Turnover': turnover,
                'Beta': beta
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {
                'Annual Return': np.nan,
                'Annual Volatility': np.nan,
                'Sharpe Ratio': np.nan,
                'Max Drawdown': np.nan,
                'Sortino Ratio': np.nan,
                'Hit Ratio': np.nan,
                'Turnover': np.nan,
                'Beta': np.nan
            }

    def save_metrics_to_latex(self, metrics_df: pd.DataFrame, save_path: str) -> None:
        """
        성과 지표를 LaTeX 형식으로 저장합니다.
        
        Args:
            metrics_df (pd.DataFrame): 성과 지표가 담긴 DataFrame
            save_path (str): 저장할 파일 경로
        """
        try:
            # 포트폴리오 카테고리 정의
            portfolio_categories = {
                'Benchmark': ['CNN Top', 'Max Sharpe'],
                'Factor': ['Factor Timing ND', 'Factor Timing FM'],
                'Momentum': ['MOM', 'STR', 'WSTR'],
                'Optimization': ['Min Variance', 'Min CVaR', 'Target 6%', 'Target 12%'],
                'Two-Stage': ['CNN Top + Max Sharpe', 'CNN Top + Min Variance', 
                            'CNN Top + Min CVaR', 'CNN Top + Target 12%']
            }

            # LaTeX 문서 시작
            latex_content = [
                "\\documentclass{article}",
                "\\usepackage{amsmath}",
                "\\usepackage{booktabs}",
                "\\usepackage{array}",
                "\\usepackage{graphicx}",
                "\\usepackage{siunitx}",
                "\\usepackage{multirow}",
                "\\pagestyle{empty}",
                "",
                "\\begin{document}",
                "",
                "\\begin{table}[h]",
                "    \\centering",
                "    \\begin{tabular}{llrrrrrrrr}",
                "    \\toprule",
                "    Category & Portfolio & E(R) & Std(R) & Sharpe Ratio & DD(R) & Sortino Ratio & \\% of +Ret & Turnover & Beta \\\\",
                "    \\midrule"
            ]

            # 각 카테고리별로 데이터 추가
            for category, portfolios in portfolio_categories.items():
                available_portfolios = [p for p in portfolios if p in metrics_df.index]
                if not available_portfolios:
                    continue

                # 카테고리의 첫 번째 포트폴리오
                first_portfolio = available_portfolios[0]
                row_data = metrics_df.loc[first_portfolio]
                latex_content.append(f"    \\multirow{{{len(available_portfolios)}}}{{*}}{{{category}}} & {first_portfolio} & {row_data['E(R)']:.4f} & {row_data['Std(R)']:.4f} & {row_data['Sharpe Ratio']:.4f} & {row_data['DD(R)']:.4f} & {row_data['Sortino Ratio']:.4f} & {row_data['% of +Ret']:.4f} & {row_data['Turnover']:.4f} & {row_data['Beta']:.4f} \\\\")

                # 나머지 포트폴리오
                for portfolio in available_portfolios[1:]:
                    row_data = metrics_df.loc[portfolio]
                    latex_content.append(f"    & {portfolio} & {row_data['E(R)']:.4f} & {row_data['Std(R)']:.4f} & {row_data['Sharpe Ratio']:.4f} & {row_data['DD(R)']:.4f} & {row_data['Sortino Ratio']:.4f} & {row_data['% of +Ret']:.4f} & {row_data['Turnover']:.4f} & {row_data['Beta']:.4f} \\\\")

                # 카테고리 구분선 추가 (마지막 카테고리 제외)
                if category != list(portfolio_categories.keys())[-1]:
                    latex_content.append("    \\midrule")

            # 테이블 종료
            latex_content.extend([
                "    \\bottomrule",
                "    \\end{tabular}",
                "    \\caption{Portfolio Performance by Category}",
                "    \\label{tab:portfolio_category}",
                "\\end{table}",
                "",
                "\\end{document}"
            ])

            # 파일 저장
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(latex_content))

            # CSV 파일도 저장
            csv_path = save_path.replace('.tex', '.csv')
            metrics_df.round(4).to_csv(csv_path, float_format='%.4f')

            self.logger.info(f"Metrics saved to LaTeX file: {save_path} and CSV file: {csv_path}")

        except Exception as e:
            self.logger.error(f"Error saving metrics to LaTeX: {str(e)}")
            raise

    def calculate_turnover(self, 
                          weights: pd.DataFrame,
                          result_dir: str,
                          model_name: str) -> pd.Series:
        """
        포트폴리오 턴오버를 계산합니다.
        
        Args:
            weights (pd.DataFrame): 포트폴리오 가중치
            result_dir (str): 결과 저장 경로
            model_name (str): 모델 이름
            
        Returns:
            pd.Series: 턴오버 시계열
        """
        weight_changes = weights.diff().abs().sum(axis=1)
        turnover = weight_changes.fillna(0).round(4)
        
        # CSV 파일 저장 (4자리수로 통일)
        csv_path = os.path.join(result_dir, f'turnover_{model_name}.csv')
        turnover.to_csv(csv_path, float_format='%.4f')

        return turnover
