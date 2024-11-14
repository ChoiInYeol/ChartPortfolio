"""
포트폴리오 최적화 프로세스의 메인 실행 파일입니다.
전체 프로세스를 조율하고 실행합니다.
"""

import os
import logging
import torch
import inquirer
from typing import List, Optional, Dict
import pandas as pd
import numpy as np

from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from utils.metrics import PerformanceMetrics
from utils.visualization import PortfolioVisualizer

def setup_logging(base_folder: str) -> logging.Logger:
    """로깅 설정을 초기화합니다."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 로그 파일 설정
    log_file = os.path.join(base_folder, 'backtest.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 콘솔 출력 설정
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 포맷 설정
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class PortfolioOptimizationPipeline:
    def __init__(self,
                 base_folder: str,
                 selections: Dict,
                 train_date: str = '2017-12-31'):
        """
        파이프라인 초기화
        
        Args:
            base_folder (str): 기본 폴더 경로
            selections (Dict): 사용자 선택 옵션
            train_date (str): 학습 시작 날짜
        """
        self.base_folder = base_folder
        self.models = selections['models']
        self.analysis_types = selections['analysis_types']
        self.plot_types = selections['plot_types']
        self.train_date = train_date
        
        # 포트폴리오 가중치와 수익률을 저장할 딕셔너리 초기화
        self.portfolio_weights = {}
        self.portfolio_returns = {}
        
        # 로깅 설정
        self.logger = setup_logging(base_folder)
        
        # 컴포넌트 초기화
        self.data_loader = DataLoader(base_folder, train_date)
        self.data_processor = DataProcessor()
        self.metrics = PerformanceMetrics()
        self.visualizer = PortfolioVisualizer()
        
        # 결과 저장 디렉토리 생성
        self.result_dir = os.path.join(base_folder, 'results')
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 옵티마이저 초기화
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        from optimization.optimizer import OptimizationManager
        self.optimizer = OptimizationManager(device)

    def run(self) -> None:
        """백테스트 프로세스를 실행합니다."""
        try:
            # 데이터 로드
            returns = self.data_loader.load_stock_returns()
            up_prob = self.data_loader.load_up_prob()
            self.logger.info(f"Data loaded - Returns: {returns.shape}, Up prob: {up_prob.shape}")
            
            self.portfolio_returns = {}
            self.portfolio_weights = {}
            
            # 1. Naive 포트폴리오 (1/N)
            # up_prob의 investment_date를 리밸런싱 날짜로 사용
            rebalance_dates = up_prob.index

            # 리밸런싱 날짜에 맞춰 1/N 가중치 설정
            naive_weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
            for date in rebalance_dates:
                if date in returns.index:
                    # 해당 시점에 존재하는 종목들에 대해서만 동일가중
                    available_stocks = returns.columns[~returns.loc[date].isna()]
                    if len(available_stocks) > 0:  # 거래 가능한 종목이 있는 경우에만 가중치 할당
                        naive_weights.loc[date, available_stocks] = 1.0 / len(available_stocks)
                        self.logger.info(f"Naive portfolio rebalancing at {date}: {len(available_stocks)} stocks")
                    else:
                        self.logger.warning(f"No available stocks for Naive portfolio at {date}")

            naive_returns, naive_net_returns = self.data_processor.calculate_portfolio_returns(
                returns, naive_weights, transaction_cost=0.00015
            )
            self.portfolio_returns['Naive'] = naive_returns
            if 'include_net' in self.plot_types:
                self.portfolio_returns['Naive (Net)'] = naive_net_returns
            self.portfolio_weights['Naive'] = naive_weights
            
            # 2. CNN 기반 벤치마크 포트폴리오
            self.logger.info("Creating CNN benchmark portfolios...")
            benchmark_weights = self.optimizer.create_benchmark_portfolios(
                up_prob=up_prob,
                returns=returns
            )
            
            # 벤치마크 매핑 정의
            benchmark_mapping = {
                'top100_equal': 'CNN Top 100',
                'bottom100_equal': 'CNN Bottom 100',
                'optimized_max_sharpe': 'CNN Top 50 + Max Sharpe',
                'optimized_min_variance': 'CNN Top 50 + Min Variance',
                'optimized_min_cvar': 'CNN Top 50 + Min CVaR'
            }
            
            # 벤치마크 포트폴리오 처리
            for internal_name, display_name in benchmark_mapping.items():
                try:
                    weights_df = benchmark_weights[internal_name]
                    returns_gross, returns_net = self.data_processor.calculate_portfolio_returns(
                        returns=returns,
                        weights=weights_df,
                        transaction_cost=0.00015
                    )
                    
                    self.portfolio_returns[display_name] = returns_gross
                    self.portfolio_weights[display_name] = weights_df
                    
                    if 'include_net' in self.plot_types:
                        self.portfolio_returns[f"{display_name} (Net)"] = returns_net
                        self.portfolio_weights[f"{display_name} (Net)"] = weights_df
                    
                    self.logger.info(f"Successfully processed benchmark portfolio: {display_name}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing benchmark portfolio {internal_name}: {str(e)}")
            
            # 3. 시계열 모델 및 CNN + 시계열 모델 포트폴리오
            for model in self.models:
                self.logger.info(f"Processing model: {model}")
                for use_prob in [False, True]:
                    weights = self.data_loader.load_model_weights(model, use_prob)
                    for name, weight_df in weights.items():
                        returns_gross, returns_net = self.data_processor.calculate_portfolio_returns(
                            returns=returns,
                            weights=weight_df,
                            transaction_cost=0.00015
                        )
                        
                        model_name = f"CNN + {name}" if use_prob else name
                        self.portfolio_returns[model_name] = returns_gross
                        self.portfolio_weights[model_name] = weight_df
                        
                        if 'include_net' in self.plot_types:
                            self.portfolio_returns[f"{model_name} (Net)"] = returns_net
                            self.portfolio_weights[f"{model_name} (Net)"] = weight_df

            # 결과 정렬을 위한 column_order 설정
            column_order = [
                'Naive',
                'CNN Top 100',
                'CNN Bottom 100',
                'CNN Top 50 + Max Sharpe',
                'CNN Top 50 + Min Variance',
                'CNN Top 50 + Min CVaR'
            ]

            # 시계열 모델 순서로 추가
            for model in self.models:
                base_name = f"{model} Top"
                model_keys = [key for key in self.portfolio_returns.keys() 
                             if key.startswith(base_name) and '(Net)' not in key]
                try:
                    n_selects = sorted(set([
                        int(key.split('Top ')[-1].split(' ')[0])
                        for key in model_keys
                    ]))
                    column_order.extend([f"{base_name} {n}" for n in n_selects])
                except Exception as e:
                    self.logger.error(f"Error processing model keys for {model}: {str(e)}")

            # CNN + 시계열 모델 순서로 추가
            for model in self.models:
                cnn_base_name = f"CNN + {model} Top"
                cnn_model_keys = [key for key in self.portfolio_returns.keys() 
                                 if key.startswith(cnn_base_name) and '(Net)' not in key]
                try:
                    cnn_n_selects = sorted(set([
                        int(key.split('Top ')[-1].split(' ')[0])
                        for key in cnn_model_keys
                    ]))
                    column_order.extend([f"{cnn_base_name} {n}" for n in cnn_n_selects])
                except Exception as e:
                    self.logger.error(f"Error processing CNN model keys for {model}: {str(e)}")

            # Net returns가 있는 경우 추가
            if 'include_net' in self.plot_types:
                net_columns = [f"{col} (Net)" for col in column_order]
                column_order.extend(net_columns)

            # 마지막에 모든 포트폴리오의 가중치를 하나의 파일로도 저장
            if 'include_net' in self.plot_types:
                all_weights_path = os.path.join(self.result_dir, 'all_portfolio_weights_with_net.csv')
            else:
                all_weights_path = os.path.join(self.result_dir, 'all_portfolio_weights.csv')

            combined_weights = pd.concat(
                {name: weights for name, weights in self.portfolio_weights.items()},
                axis=1,
                names=['Portfolio', 'Stock']
            )
            combined_weights.to_csv(all_weights_path)
            self.logger.info(f"All portfolio weights saved to {all_weights_path}")

            # 디버깅을 위한 출력
            self.logger.info("Final column order:")
            for col in column_order:
                self.logger.info(f"- {col}")

            # 존재하는 열만 선택하여 DataFrame 생성
            portfolio_returns_df = pd.DataFrame({
                col: self.portfolio_returns[col] 
                for col in column_order 
                if col in self.portfolio_returns
            })
            
            print(portfolio_returns_df.head())
            
            # 선택된 분석 수행
            if 'performance_metrics' in self.analysis_types:
                self._analyze_performance(portfolio_returns_df)
            
            # 포트폴리오 비교 시각화 추가
            self.visualizer.plot_portfolio_comparison(
                returns_dict=portfolio_returns_df,
                title="Portfolio Performance Comparison",
                result_dir=self.result_dir,
                include_net=False  # 기본 버전
            )
            if 'include_net' in self.plot_types:
                self.visualizer.plot_portfolio_comparison(
                    returns_dict=portfolio_returns_df,
                    title="Portfolio Performance Comparison (with Net Returns)",
                    result_dir=self.result_dir,
                    include_net=True  # Net returns 포함 버전
                )

            # 롤링 메트릭스
            if 'rolling_metrics' in self.analysis_types:
                self.visualizer.plot_rolling_metrics(
                    portfolio_returns_df.pct_change(fill_method=None),
                    window=252,
                    result_dir=self.result_dir,
                    include_net=False  # 기본 버전
                )
                if 'include_net' in self.plot_types:
                    self.visualizer.plot_rolling_metrics(
                        portfolio_returns_df.pct_change(fill_method=None),
                        window=252,
                        result_dir=self.result_dir,
                        include_net=True  # Net returns 포함 버전
                    )

            # Drawdown 분석
            if 'drawdown_analysis' in self.analysis_types:
                self.visualizer.plot_drawdown_analysis(
                    portfolio_returns_df,
                    self.result_dir,
                    include_net=False  # 기본 버전
                )
                if 'include_net' in self.plot_types:
                    self.visualizer.plot_drawdown_analysis(
                        portfolio_returns_df,
                        self.result_dir,
                        include_net=True  # Net returns 포함 버전
                    )

            if 'turnover_analysis' in self.analysis_types:
                # 모든 포트폴리오의 turnover를 한 번에 분석
                self.visualizer.plot_turnover_analysis(
                    weights_dict=self.portfolio_weights,
                    result_dir=self.result_dir
                )
            
            # 포트폴리오 비중 변화 시각화
            if 'weight_evolution' in self.plot_types:
                for name, weight_df in self.portfolio_weights.items():
                    # area plot 저장
                    self.visualizer.plot_weight_evolution(
                        weight_df,
                        self.result_dir,
                        name,
                        plot_type='area'
                    )
                    # bar plot 저장 
                    self.visualizer.plot_weight_evolution(
                        weight_df,
                        self.result_dir,
                        name,
                        plot_type='bar'
                    )
            
            self.logger.info("Backtest completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during backtest: {str(e)}", exc_info=True)
            raise

    def _analyze_performance(self, returns: pd.DataFrame) -> None:
        """성과 분석을 수행하고 저장합니다."""
        metrics_dict = {}
        for column in returns.columns:
            # fill_method=None으로 명시적 지정
            metrics = self.metrics.calculate_portfolio_metrics(
                returns[column].pct_change(fill_method=None).dropna(),
                weights=self.portfolio_weights.get(column)  # 클래스 속성 사용
            )
            metrics_dict[column] = metrics
        
        metrics_df = pd.DataFrame(metrics_dict).T
        
        # 성과 지표 저장
        self.metrics.save_metrics_latex(
            metrics_df,
            self.result_dir,
            'all_portfolios'
        )

def main():
    """메인 실행 함수"""
    base_folder = os.path.dirname(os.path.abspath(__file__))
    
    # 사용자 선택 옵션
    questions = [
        inquirer.Checkbox('models',
                         message="Select model type(s)",
                         choices=['GRU', 'TCN', 'TRANSFORMER'],
                         default=['GRU']),
        
        inquirer.Checkbox('analysis_types',
                         message="Select analysis type(s)",
                         choices=['performance_metrics', 'rolling_metrics', 
                                'drawdown_analysis', 'turnover_analysis'],
                         default=['performance_metrics']),
        
        inquirer.Checkbox('plot_types',
                         message="Select plot type(s)",
                         choices=['weight_evolution', 'include_net'],
                         default=['weight_evolution'])
    ]
    
    selections = inquirer.prompt(questions)
    
    if not selections['models']:
        print("At least one model must be selected")
        return
    
    # 선택 사항 출력
    logger = logging.getLogger(__name__)
    logger.info(f"Selected models: {selections['models']}")
    logger.info(f"Selected analysis types: {selections['analysis_types']}")
    logger.info(f"Selected plot types: {selections['plot_types']}")
    
    # 파이프라인 실행
    pipeline = PortfolioOptimizationPipeline(
        base_folder=base_folder,
        selections=selections,
        train_date='2017-12-31'
    )
    pipeline.run()

if __name__ == "__main__":
    main()