"""
포트폴리오 최적화 프로세스의 메인 실행 파일입니다.
전체 프로세스를 조율하고 실행합니다.
"""

import os
import logging
import torch
from typing import List, Optional
import pandas as pd

from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from optimization.optimizer import OptimizationManager
from utils.metrics import PerformanceMetrics
from utils.visualization import PortfolioVisualizer

class PortfolioOptimizationPipeline:
    """
    포트폴리오 최적화 파이프라인을 관리하는 클래스입니다.
    """
    
    def __init__(self,
                 base_folder: str,
                 models: List[str] = None,
                 model_window_sizes: List[int] = None,
                 optimization_window_size: int = 60,
                 train_date: str = '2017-12-31'):
        """
        파이프라인 초기화
        
        Args:
            base_folder (str): 기본 폴더 경로
            models (List[str]): 사용할 모델 리스트
            model_window_sizes (List[int]): CNN 모델의 윈도우 크기 리스트
            optimization_window_size (int): 포트폴리오 최적화를 위한 윈도우 크기
            train_date (str): 학습 시작 날짜
        """
        self.base_folder = base_folder
        self.models = models or ['CNN']
        self.model_window_sizes = model_window_sizes or [20]
        self.optimization_window_size = optimization_window_size
        self.train_date = train_date
        
        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 로깅 설정
        self._setup_logging()
        
        # 컴포넌트 초기화
        self.data_loader = DataLoader(base_folder, train_date)
        self.data_processor = DataProcessor()
        self.optimizer = OptimizationManager(self.device)
        self.metrics = PerformanceMetrics()
        self.visualizer = PortfolioVisualizer()
        
        self.logger.info(f"Using device: {self.device}")

    def _setup_logging(self) -> None:
        """
        로깅 설정을 초기화합니다.
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.base_folder, 'optimization.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def process_model_window(self,
                           model: str,
                           model_window_size: int,
                           us_ret: pd.DataFrame,
                           benchmark_data: pd.DataFrame) -> Optional[str]:
        """
        특정 모델과 윈도우 크기에 대한 최적화를 수행합니다.
        
        Args:
            model (str): 모델 이름
            model_window_size (int): CNN 모델의 윈도우 크기
            us_ret (pd.DataFrame): 주식 수익률 데이터
            benchmark_data (pd.DataFrame): 벤치마크 데이터
            
        Returns:
            Optional[str]: 처리 결과 메시지
        """
        self.logger.info(f'Processing model: {model}, model window size: {model_window_size}')
        
        # 결과 저장 폴더 생성
        folder_name = os.path.join(self.base_folder, 'WORK_DIR', f'{model}{model_window_size}')
        os.makedirs(folder_name, exist_ok=True)
        
        try:
            # 앙상블 결과 로드
            ensemble_results = self.data_loader.load_ensemble_results(model, model_window_size)
            if ensemble_results.empty:
                return None
            
            # 주식 선택 및 포트폴리오 구성
            n_stocks_100 = 100
            n_stocks_10 = round(us_ret.shape[1] / 10)
            
            selected_stocks = self.data_processor.process_portfolio_stocks(
                ensemble_results, n_stocks_100
            )
            
            # 포트폴리오 수익률 계산
            portfolio_ret = {}
            rebalance_dates = {}
            valid_stock_ids = set(us_ret.columns)
            
            for portfolio_name, stocks in selected_stocks.items():
                portfolio_ret[portfolio_name], rebalance_dates[portfolio_name] = \
                    self.data_processor.calculate_portfolio_returns(
                        us_ret, stocks, valid_stock_ids
                    )
            
            # 기본 포트폴리오 추가
            portfolio_ret['Naive'] = (1 + us_ret.mean(axis=1)).cumprod()
            portfolio_ret['SPY'] = benchmark_data['Cumulative Returns']
            portfolio_ret = pd.DataFrame(portfolio_ret)
            
            # 포트폴리오 최적화
            optimized_portfolios = {}
            portfolio_weights = {}
            
            for method in ['max_sharpe',
                           #'min_variance',
                           #'min_cvar'
                           ]:
                optimized_returns, weights = self.optimizer.process_portfolio(
                    ensemble_results, 
                    us_ret, 
                    method, 
                    model, 
                    model_window_size,
                    self.base_folder,
                    optimization_window_size=self.optimization_window_size
                )
                optimized_portfolios[method] = optimized_returns
                portfolio_weights[method] = weights
            
            # 결과 시각화
            self.visualizer.plot_optimized_portfolios(
                portfolio_ret, optimized_portfolios, model, model_window_size,
                folder_name, rebalance_dates
            )
            
            # 성과 지표 계산 및 저장
            combined_returns = portfolio_ret.copy()
            for method, returns in optimized_portfolios.items():
                combined_returns[f'Optimized_{method}'] = (1 + returns).cumprod()
            
            performance_metrics = self.metrics.calculate_metrics(
                combined_returns, portfolio_weights, selected_stocks, n_stocks_10
            )
            self.metrics.save_metrics(performance_metrics, model, model_window_size, folder_name)
            
            # # 추가 시각화
            # for method, weights in portfolio_weights.items():
            #     self.visualizer.plot_weight_distribution(
            #         weights, model, model_window_size, folder_name
            #     )
            #     self.visualizer.plot_turnover_analysis(
            #         weights, model, model_window_size, folder_name
            #     )
            
            return f"Completed processing for {model} with model window size {model_window_size}"
            
        except Exception as e:
            self.logger.error(f"Error processing {model}{model_window_size}: {str(e)}")
            return None

    def run(self) -> None:
        """
        전체 최적화 프로세스를 실행합니다.
        """
        self.logger.info("Starting portfolio optimization process")
        
        # 데이터 로드
        us_ret = self.data_loader.load_stock_returns()
        self.logger.info("Stock returns data loaded")
        
        benchmark_data = self.data_loader.load_benchmark()
        self.logger.info("Benchmark data loaded")
        
        # 각 모델과 윈도우 크기에 대해 처리
        for model in self.models:
            for model_window_size in self.model_window_sizes:
                result = self.process_model_window(
                    model, model_window_size, us_ret, benchmark_data
                )
                if result:
                    self.logger.info(result)
        
        self.logger.info("Portfolio optimization process completed")

def main():
    """
    메인 실행 함수
    """
    base_folder = os.path.dirname(os.path.abspath(__file__))
    
    pipeline = PortfolioOptimizationPipeline(
        base_folder=base_folder,
        models=['CNN'],
        model_window_sizes=[20],
        optimization_window_size=60,
        train_date='2017-12-31'
    )
    pipeline.run()

if __name__ == "__main__":
    main()