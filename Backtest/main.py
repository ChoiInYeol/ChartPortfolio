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

from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from utils.metrics import PerformanceMetrics
from utils.visualization import PortfolioVisualizer
from optimization.optimizer import OptimizationManager

def setup_logging(base_folder: str) -> logging.Logger:
    """로깅 설정을 초기화합니다."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
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
    def __init__(self,
                 base_folder: str,
                 selections: Dict,
                 train_date: str = '2017-12-31'):
        """
        백테스트 초기화
        
        Args:
            base_folder (str): 기본 폴더 경로
            selections (Dict): 사용자 선택 옵션
            train_date (str): 학습 시작 날짜
        """
        self.base_folder = base_folder
        self.models = selections['models']
        self.train_date = train_date
        
        # 결과 저장 디렉토리 생성
        self.result_dir = os.path.join(base_folder, 'results')
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 로깅 설정
        self.logger = setup_logging(base_folder)
        
        # 컴포넌트 초기화
        self.data_loader = DataLoader(base_folder, train_date)
        self.data_processor = DataProcessor()
        self.metrics = PerformanceMetrics()
        self.visualizer = PortfolioVisualizer()
        self.optimizer = OptimizationManager()

    def run(self) -> None:
        """백테스트를 실행합니다."""
        try:
            # 1. 데이터 로드
            returns = self.data_loader.load_stock_returns()
            up_prob = self.data_loader.load_up_prob()
            self.logger.info(f"Data loaded - Returns: {returns.shape}, Up prob: {up_prob.shape}")
            
            # 2. 포트폴리오 구성
            portfolio_weights = {}
            
            # 2.1 Naive 포트폴리오 (1/N)
            naive_weights = pd.DataFrame(1.0/len(returns.columns), 
                                       index=returns.index, 
                                       columns=returns.columns)
            portfolio_weights['Naive'] = naive_weights
            
            # 2.2 CNN 기반 벤치마크 포트폴리오
            benchmark_weights = self.optimizer.create_benchmark_portfolios(up_prob, returns)
            portfolio_weights.update(benchmark_weights)
            
            # 2.3 시계열 모델 포트폴리오
            for model in self.models:
                model_weights = self.data_loader.load_model_weights(model, use_prob=True)
                portfolio_weights.update(model_weights)
            
            # 3. 수익률 계산
            portfolio_returns = {}
            for name, weights in portfolio_weights.items():
                returns_series = self.data_processor.calculate_portfolio_returns(
                    returns=returns,
                    weights=weights
                )
                portfolio_returns[name] = returns_series
            
            # 4. 성과 지표 계산
            metrics_results = {}
            for name, rets in portfolio_returns.items():
                metrics = self.metrics.calculate_portfolio_metrics(
                    returns=rets,
                    weights=portfolio_weights[name]
                )
                metrics_results[name] = metrics
            
            # 결과를 DataFrame으로 변환
            metrics_df = pd.DataFrame(metrics_results).T
            
            # 5. 결과 저장
            # 5.1 성과 지표 저장
            self.metrics.save_metrics_latex(metrics_df, self.result_dir, 'all_portfolios')
            
            # 5.2 수익률 그래프 저장
            returns_df = pd.DataFrame(portfolio_returns)
            self.visualizer.plot_portfolio_comparison(
                returns_dict=returns_df,
                title="Portfolio Performance Comparison",
                result_dir=self.result_dir
            )
            
            self.logger.info("Backtest completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during backtest: {str(e)}", exc_info=True)
            raise

def main():
    """메인 실행 함수"""
    base_folder = os.path.dirname(os.path.abspath(__file__))
    
    # 사용자 선택 옵션
    questions = [
        inquirer.Checkbox('models',
                         message="Select model type(s)",
                         choices=['GRU', 'TCN', 'TRANSFORMER'],
                         default=['GRU'])
    ]
    
    selections = inquirer.prompt(questions)
    
    if not selections['models']:
        print("At least one model must be selected")
        return
    
    # 선택 사항 출력
    logger = logging.getLogger(__name__)
    logger.info(f"Selected models: {selections['models']}")
    
    # 백테스트 실행
    backtest = PortfolioBacktest(
        base_folder=base_folder,
        selections=selections,
        train_date='2017-12-31'
    )
    backtest.run()

if __name__ == "__main__":
    main()