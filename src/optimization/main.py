"""
포트폴리오 최적화 프로세스의 메인 실행 파일입니다.
전체 프로세스를 조율하고 실행합니다.
"""

import os
import logging
import inquirer
from typing import Dict
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
import argparse

from data_loader import DataLoader
from utils.metrics import PerformanceMetrics
from utils.visualization import PortfolioVisualizer
from src.optimization.optimizer import OptimizationManager  # 팩터 타이밍 기능이 통합된 매니저 사용

def setup_logging(base_folder: str) -> logging.Logger:
    """로깅 설정을 초기화합니다."""
    logger = logging.getLogger(__name__)
    
    # 이미 핸들러가 있다면 제거
    if logger.handlers:
        logger.handlers.clear()
    
    logger.setLevel(logging.INFO)
    
    # 로그 디렉토리 생성
    os.makedirs(base_folder, exist_ok=True)
    log_file = os.path.join(base_folder, 'backtest.log')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class PortfolioBacktest:
    """포트폴리오 백테스트를 수행하는 클래스입니다."""
    
    def __init__(self,
                 base_folder: str,
                 data_size: int,
                 result_dir: str = 'results',
                 use_cache: bool = True) -> None:
        """
        PortfolioBacktest를 초기화합니다.
        
        Args:
            base_folder (str): 데이터가 저장된 기본 폴더 경로
            data_size (int): 데이터 크기
            result_dir (str): 결과를 저장할 디렉토리 경로
            use_cache (bool): 캐시 사용 여부
        """
        self.base_folder = base_folder
        self.data_size = data_size
        self.result_dir = result_dir
        self.use_cache = use_cache
        
        # 로깅 설정
        self.logger = setup_logging(result_dir)
        
        # 성능 측정 및 시각화 객체 초기화
        self.metrics = PerformanceMetrics()
        self.visualizer = PortfolioVisualizer()
        self.optimizer = OptimizationManager(result_dir=self.result_dir)
        
        # 리밸런싱 날짜 초기화
        self.rebalance_dates = None

    def _determine_rebalancing_freq(self) -> str:
        """리밸런싱 주기를 결정합니다."""
        return 'ME'  # 월말 리밸런싱으로 고정

    def _get_rebalancing_dates(self, returns: pd.DataFrame, freq: str = 'ME') -> pd.DatetimeIndex:
        """
        리밸런싱 날짜를 계산합니다.
        
        Args:
            returns (pd.DataFrame): 수익률 데이터
            freq (str): 리밸런싱 주기 (예: 'ME' - 월말)
            
        Returns:
            pd.DatetimeIndex: 리밸런싱 날짜
        """
        # 리밸런싱 날짜 계산 (월말 기준)
        rebalance_dates = pd.date_range(
            start=returns.index[0],
            end=returns.index[-1],
            freq=freq
        )
        
        # 실제 거래일에 맞춰 리밸런싱 날짜 조정
        adjusted_dates = []
        for date in rebalance_dates:
            month_end = date
            available_dates = returns.index[returns.index <= month_end]
            if len(available_dates) > 0:
                adjusted_dates.append(available_dates[-1])
        
        return pd.DatetimeIndex(sorted(set(adjusted_dates)))

    def create_benchmark_portfolios(self, up_prob: pd.DataFrame, returns: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """벤치마크 포트폴리오들을 생성합니다."""
        # 기존 벤치마크 포트폴리오 생성
        benchmark_weights = self.optimizer.create_benchmark_portfolios(
            up_prob=up_prob,
            returns=returns,
            rebalance_dates=self.rebalance_dates,
            use_cache=True
        )
        
        # Target Return 포트폴리오 추가
        target_returns = [0.06, 0.08, 0.10, 0.12]  # 연간 목표 수익률
        for target in target_returns:
            try:
                # 일반 Target Return 포트폴리오
                weights_df = pd.DataFrame(0.0, index=self.rebalance_dates, columns=returns.columns)
                
                for date in self.rebalance_dates:
                    try:
                        # 과거 60일 수익률 데이터 사용
                        historical_returns = returns.loc[:date].tail(60)
                        
                        # Target Return 최적화
                        weights = self.optimizer.optimize_portfolio_with_gurobi(
                            returns=historical_returns,
                            method='target_return',
                            target_return=target,
                            show_progress=False
                        )
                        weights_df.loc[date] = weights
                        
                    except Exception as e:
                        self.logger.error(f"Error in Target Return {target*100:.0f}% at {date}: {str(e)}")
                        # 이전 가중치 사용 또는 동일가중
                        prev_dates = weights_df.index[weights_df.index < date]
                        if len(prev_dates) > 0:
                            weights_df.loc[date] = weights_df.loc[prev_dates[-1]]
                        else:
                            weights_df.loc[date] = 1.0 / len(returns.columns)
                
                benchmark_weights[f'Target {target*100:.0f}%'] = weights_df
                
                # CNN Top + Target Return 포트폴리오
                cnn_weights_df = pd.DataFrame(0.0, index=self.rebalance_dates, columns=returns.columns)
                
                for date in self.rebalance_dates:
                    try:
                        # CNN이 선택한 종목들 찾기
                        prob_dates = up_prob.index[up_prob.index <= date]
                        if len(prob_dates) == 0:
                            continue
                        
                        prob_date = prob_dates[-1]
                        month_probs = up_prob.loc[prob_date]
                        threshold = month_probs.median()
                        up_stocks = month_probs[month_probs >= threshold].index
                        
                        if len(up_stocks) == 0:
                            continue
                        
                        # CNN이 선택한 종목들에 대해서만 Target Return 최적화
                        historical_returns = returns.loc[:date].tail(60)[up_stocks]
                        
                        weights = self.optimizer.optimize_portfolio_with_gurobi(
                            returns=historical_returns,
                            method='target_return',
                            target_return=target,
                            show_progress=False
                        )
                        cnn_weights_df.loc[date, up_stocks] = weights
                        
                    except Exception as e:
                        self.logger.error(f"Error in CNN Top + Target {target*100:.0f}% at {date}: {str(e)}")
                        # 이전 가중치 사용 또는 동일가중
                        prev_dates = cnn_weights_df.index[cnn_weights_df.index < date]
                        if len(prev_dates) > 0:
                            cnn_weights_df.loc[date] = cnn_weights_df.loc[prev_dates[-1]]
                        else:
                            cnn_weights_df.loc[date, up_stocks] = 1.0 / len(up_stocks)
                
                benchmark_weights[f'CNN Top + Target {target*100:.0f}%'] = cnn_weights_df
                
            except Exception as e:
                self.logger.error(f"Error creating Target Return {target*100:.0f}% portfolios: {str(e)}")
        
        return benchmark_weights

    def run(self) -> None:
        """백테스트를 실행합니다."""
        try:
            # 1. 데이터 로드
            print("Loading data...")
            loader = DataLoader(base_folder=self.base_folder, data_size=self.data_size)
            returns = loader.load_returns()
            probs = loader.load_probabilities()
            
            # 2. 리밸런싱 날짜 설정
            rebalance_freq = self._determine_rebalancing_freq()
            self.rebalance_dates = self._get_rebalancing_dates(returns, rebalance_freq)
            
            # 3. 포트폴리오 생성
            portfolio_weights = {}
            
            # 3.1 팩터 타이밍 포트폴리오 생성
            print("Creating factor timing portfolios...")
            try:
                factor_scores = {
                    'rising_prob': probs,
                    'momentum': returns.rolling(window=60).mean()
                }
                factor_timing_weights = self.optimizer.create_factor_timing_portfolios(
                    returns=returns,
                    factor_scores=factor_scores,
                    rebalance_freq=rebalance_freq,
                    top_pct=0.3,
                    fm_window=12,
                    lookback_period=60,
                    min_history_months=6,
                    use_cache=self.use_cache
                )
                portfolio_weights.update(factor_timing_weights)
                
            except Exception as e:
                self.logger.error(f"Error creating factor timing portfolios: {str(e)}")
            
            # 3.2 벤치마크 포트폴리오 생성
            print("Creating benchmark portfolios...")
            benchmark_weights = self.create_benchmark_portfolios(probs, returns)
            portfolio_weights.update(benchmark_weights)
            
            # 4. 포트폴리오 수익률 계산
            print("Calculating portfolio returns...")
            portfolio_returns = {}
            for name, weights in portfolio_weights.items():
                try:
                    returns_series = self.metrics.calculate_portfolio_returns(returns, weights)
                    returns_series.name = name
                    portfolio_returns[name] = returns_series
                except Exception as e:
                    self.logger.error(f"Error calculating returns for {name}: {str(e)}")
            
            # 5. 성과 측정
            print("Calculating performance metrics...")
            try:
                metrics_df = self.metrics.calculate_portfolio_metrics(portfolio_returns)
                
                # 5.1 성과 지표 저장
                metrics_dir = os.path.join(self.result_dir, 'metrics')
                os.makedirs(metrics_dir, exist_ok=True)
                
                metrics_path = os.path.join(metrics_dir, 'portfolio_metrics.csv')
                metrics_df.to_csv(metrics_path)
                self.logger.info(f"Performance metrics saved to {metrics_path}")
                
                latex_path = os.path.join(metrics_dir, 'portfolio_metrics.tex')
                self.metrics.save_metrics_to_latex(metrics_df, latex_path)
                self.logger.info(f"LaTeX metrics saved to {latex_path}")
                
            except Exception as e:
                self.logger.error(f"Error calculating metrics: {str(e)}")
            
            # 6. 시각화
            print("Creating visualizations...")
            try:
                plots_dir = os.path.join(self.result_dir, 'plots')
                os.makedirs(plots_dir, exist_ok=True)
                
                # 누적 수익률 플롯
                cum_returns_path = os.path.join(plots_dir, 'cumulative_returns.png')
                self.visualizer.plot_cumulative_returns(portfolio_returns, cum_returns_path)
                self.logger.info(f"Cumulative returns plot saved to {cum_returns_path}")
                
            except Exception as e:
                self.logger.error(f"Error creating visualizations: {str(e)}")
            
            print("Backtest completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Error in backtest: {str(e)}")
            raise

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Portfolio Backtest')
    parser.add_argument('--mode', type=str, default='Full Process',
                       help='Process mode (Full Process or Analysis Only)')
    parser.add_argument('--size', type=int, default=500,
                       help='Data size (number of stocks)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable optimization cache')
    
    args = parser.parse_args()
    
    # 설정 로깅
    logger = logging.getLogger(__name__)
    logger.info(f"Selected mode: {args.mode}")
    logger.info(f"Selected data size: {args.size}")
    logger.info(f"Using optimization cache: {not args.no_cache}")
    
    # 기본 폴더 설정
    base_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result_dir = os.path.join(base_folder, 'Backtest', 'results', f'size_{args.size}')
    
    # 백테스트 실행
    backtest = PortfolioBacktest(
        base_folder=base_folder,
        data_size=args.size,
        result_dir=result_dir,
        use_cache=not args.no_cache
    )
    backtest.run()

if __name__ == '__main__':
    main()