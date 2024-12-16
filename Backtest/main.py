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

from data_loader import DataLoader
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
                 data_size: int,
                 selections: Dict,
                 train_date: str = '2017-12-31',
                 end_date: str = '2024-07-05'):
        """
        백테스트 초기화
        
        Args:
            base_folder (str): 기본 폴더 경로
            data_size (int): 데이터 크기 (50, 370, 500, 2055)
            selections (Dict): 사용자 선택 옵션
            train_date (str): 학습 시작 날짜
            end_date (str): 투자 종료 날짜
        """
        self.base_folder = base_folder
        self.data_size = data_size
        self.models = selections['models']
        self.train_date = train_date
        self.end_date = end_date
        
        # 로깅 설정
        self.logger = setup_logging(base_folder)
        
        # 결과 디렉토리 설정
        self.result_dir = os.path.join(base_folder, 'results', f'size_{data_size}')
        os.makedirs(self.result_dir, exist_ok=True)
        
        # figures 디렉토리 생성
        figures_dir = os.path.join(self.result_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        # weights 디렉토리 생성
        weights_dir = os.path.join(self.result_dir, 'weights')
        os.makedirs(weights_dir, exist_ok=True)
        
        # 데이터 로더 초기화
        self.data_loader = DataLoader(
            base_folder=base_folder,
            data_size=data_size,
            train_date=train_date,
            end_date=end_date
        )
        
        # 컴포넌트 초기화
        self.metrics = PerformanceMetrics()
        self.visualizer = PortfolioVisualizer()
        self.optimizer = OptimizationManager()

    def _determine_rebalancing_freq(self) -> str:
        """
        ws와 pw 값에 따라 적절한 리밸런싱 주기를 결정합니다.
        
        Returns:
            str: 리밸런싱 주기 ('M', 'Q', 'Y')
        """
        # 모든 경우에 대해 월단위 리밸런싱 사용
        # 매월 초에 매수, 매월 말에 매도
        return 'ME'

    def run(self) -> None:
        """백테스트를 실행합니다."""
        try:
            # 1. 데이터 로드
            data = self.data_loader.load_data()
            returns = data['returns']
            probs = data['probs']
            benchmark_returns = self.data_loader.load_benchmark()
            self.logger.info(f"Data loaded - Returns: {returns.shape}, Up prob: {probs.shape}")
            
            # 2. 포트폴리오 구성
            portfolio_weights = {}
            
            # 리밸런싱 날짜 계산
            rebalance_dates = pd.date_range(
                start=returns.index[0],
                end=returns.index[-1],
                freq='ME'
            )
            
            # 2.1 Naive 포트폴리오 (1/N)
            naive_weights = pd.DataFrame(
                index=rebalance_dates,
                columns=returns.columns,
                data=1.0/len(returns.columns),
                dtype=np.float64
            )
            portfolio_weights['Naive'] = naive_weights
            
            # 2.2 CNN 기반 벤치마크 포트폴리오
            print("Creating CNN benchmark portfolios...")
            N = 500
            result_dir = os.path.join(self.base_folder, 'results')
            benchmark_weights = self.optimizer.create_benchmark_portfolios(
                probs, 
                returns, 
                N=N,
                result_dir=result_dir,
                rebalance_dates=rebalance_dates  # 리밸런싱 날짜 전달
            )
            portfolio_weights.update(benchmark_weights)
            print("Completed creating CNN benchmark portfolios.")
            
            # 2.3 시계열 모델 트폴리오
            print(f"Loading weights for {self.models} models...")
            for model in self.models:
                try:
                    model_weights = self.data_loader.load_ts_model_weights(model)
                    if model_weights:  # 빈 딕셔너리가 아닌 경우에만 업데이트
                        portfolio_weights.update(model_weights)
                        print(f"{model} weights updated")
                except Exception as e:
                    self.logger.warning(f"Failed to load weights for {model}: {str(e)}")
                    continue
            print(f"Completed loading weights for {len(portfolio_weights)} portfolios.")
            
            # 2.4 포트폴리오 가중치 저장
            weights_dir = os.path.join(self.result_dir, 'weights')
            os.makedirs(weights_dir, exist_ok=True)
            
            for name, weights in portfolio_weights.items():
                # 파일명에 사용할 수 없는 문자 제거
                safe_name = "".join(x for x in name if x.isalnum() or x in ['-', '_'])
                save_path = os.path.join(weights_dir, f'{safe_name}_weights.csv')
                weights.to_csv(save_path)
                self.logger.info(f"Saved {name} weights to {save_path}")
            
            # 3. 수익률 계산
            portfolio_returns = {}
            for name, weights in portfolio_weights.items():
                # 리밸런싱 주기를 고려한 수익률 계산
                returns_series = self.metrics.calculate_portfolio_returns(
                    returns=returns,
                    weights=weights,
                    rebalancing_freq='ME'  # 월말 리밸런싱
                )
                returns_series.name = name  # 시리즈에 이름 부여
                portfolio_returns[name] = returns_series
            
            # 4. 성과 지표 계산 및 저장
            metrics_results = {}
            for name, rets in portfolio_returns.items():
                metrics = self.metrics.calculate_portfolio_metrics(
                    returns=rets,
                    weights=portfolio_weights[name],
                    benchmark_returns=benchmark_returns,
                    result_dir=self.result_dir  # 결과 저장 경로 전달
                )
                metrics_results[name] = metrics
            
            # 결과를 DataFrame으로 변환
            metrics_df = pd.DataFrame(metrics_results).T
            
            # 5. 결과 저장 및 시각화
            # 5.1 성과 지표 저장 (CSV, LaTeX)
            metrics_dir = os.path.join(self.result_dir, 'metrics')
            os.makedirs(metrics_dir, exist_ok=True)
            
            metrics_df.to_csv(os.path.join(metrics_dir, 'all_portfolios_metrics.csv'))
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
            }).to_latex(os.path.join(metrics_dir, 'all_portfolios_metrics.tex'))
            
            # 5.2 수익률 그래프 저장
            returns_df = pd.DataFrame(portfolio_returns)
            
            # 실제 컬럼명 확인을 위한 로깅 추가
            self.logger.info(f"Available portfolios: {returns_df.columns.tolist()}")

            selected_portfolios = [
                'Naive',
                'CNN Top',
                
                'Max Sharpe',
                'Min Variance',
                'Min CVaR',
                
                'CNN Top + Max Sharpe',
                'CNN Top + Min Variance',
                'CNN Top + Min CVaR',  # 'Cvar'를 'CVaR'로 수정
                
                'GRU',
                'TCN',
                'TRANSFORMER',
                
                'CNN + GRU',
                'CNN + TCN',
                'CNN + TRANSFORMER'
            ]

            # 선택된 포트폴리오 존재 여부 확인
            missing_portfolios = [p for p in selected_portfolios if p not in returns_df.columns]
            if missing_portfolios:
                self.logger.warning(f"Missing portfolios: {missing_portfolios}")
                # 존재하는 포트폴리오만 선택
                selected_portfolios = [p for p in selected_portfolios if p in returns_df.columns]

            # 선택된 포트폴리오만 필터
            selected_returns = returns_df[selected_portfolios]

            # 수익률 그래프 저장
            self.visualizer.plot_portfolio_comparison(
                returns_dict=selected_returns,
                title="Portfolio Performance Comparison",
                result_dir=self.result_dir,
                selected_portfolios=selected_portfolios
            )
            
            # 5.3 포트폴리오 가중치 저장
            for name, weights in portfolio_weights.items():
                safe_name = "".join(x for x in name if x.isalnum() or x in ['-', '_'])
                self.visualizer.plot_weights(
                    weights=weights,
                    save_path=os.path.join(self.result_dir, 'figures', f'{safe_name}_weights_plot.png'),
                    title=f"{name} Portfolio Weights"
                )
            
            # 5.3 전략 비교 그래프 저장
            self.visualizer.plot_strategy_comparison(
                metrics_df=metrics_df,
                result_dir=self.result_dir,
                title="Strategy Performance Comparison"
            )
            
            self.logger.info("Backtest completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during backtest: {str(e)}", exc_info=True)
            raise

def main():
    """메인 실행 함수"""
    base_folder = os.path.dirname(os.path.abspath(__file__))
    
    # 로깅 설정
    logger = setup_logging(base_folder)
    
    # 설정 파일 로드
    with open(os.path.join(base_folder, 'weight.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    
    # 경로 설정 수정
    paths = config['paths']
    paths['results'] = os.path.join(base_folder, 'results')  # Backtest/results로 수정
    paths['ts_model_data'] = os.path.join(base_folder, '..', 'TS_Model', 'data')  # 상대 경로 수정
    portfolio_config = config['portfolios']
    
    # 사용자 선택 옵션
    questions = [
        inquirer.List('mode',
                     message="Select operation mode",
                     choices=['Full Process', 'Visualization Only'],
                     default='Full Process'),
        inquirer.List('data_size',
                     message="Select data size",
                     choices=config['base_settings']['data_sizes']),
        inquirer.List('window_size',
                     message="Select window size",
                     choices=[20, 60, 120],
                     default=60),
        inquirer.List('prediction_window',
                     message="Select prediction window",
                     choices=[20, 60, 120],
                     default=60),
        inquirer.Checkbox('models',
                         message="Select model type(s)",
                         choices=['GRU', 'TCN', 'TRANSFORMER'],
                         default=['GRU'])
    ]
    
    selections = inquirer.prompt(questions)
    
    if selections['mode'] == 'Visualization Only':
        # 시각화만 실행
        logger.info("Starting Visualization Only mode...")
        
        # 파일명 -> 표시명 매핑 먼저 정의
        name_mapping = portfolio_config['names']
        
        data_loader = DataLoader(
            base_folder=base_folder,
            data_size=selections['data_size'],
            train_date=config['base_settings']['train_date'],
            end_date=config['base_settings']['end_date'],
            ws=selections['window_size'],
            pw=selections['prediction_window']
        )
        result_dir = data_loader.result_dir
        logger.info(f"Result directory: {result_dir}")
        
        # 필요한 데이터 로드
        try:
            data = data_loader.load_data()
            returns_df = data['returns']
            up_prob = data['probs']
            logger.info(f"Data loaded - Returns shape: {returns_df.shape}, Probs shape: {up_prob.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return
        
        # 포트폴리오 가중치 저장할 딕셔너리
        portfolio_weights = {}
        
        # 1. TS_Model weights 로드
        logger.info(f"Loading weights for {selections['models']} models...")
        for model in selections['models']:
            try:
                model_weights = data_loader.load_ts_model_weights(model)
                if model_weights:  # 빈 딕셔너리가 아닌 경우에만 업데이트
                    portfolio_weights.update(model_weights)
                    # 가중치 파일 저장
                    weights_dir = os.path.join(result_dir, 'weights')
                    os.makedirs(weights_dir, exist_ok=True)
                    for name, weights in model_weights.items():
                        safe_name = "".join(x for x in name if x.isalnum() or x in ['-', '_'])
                        save_path = os.path.join(weights_dir, f'{safe_name}_weights.csv')
                        weights.to_csv(save_path)
                        logger.info(f"Saved {name} weights to {save_path}")
                    logger.info(f"{model} weights loaded and saved")
            except Exception as e:
                logger.warning(f"Failed to load weights for {model}: {str(e)}")
                continue
        
        # 2. 저장된 가중치 데이터 로드
        weights_dir = os.path.join(result_dir, paths['weights'])
        if os.path.exists(weights_dir):
            weight_files = os.listdir(weights_dir)
            logger.info(f"Found {len(weight_files)} weight files in {weights_dir}")
            
            for file in weight_files:
                if file.endswith('_weights.csv'):
                    file_name = file.replace('_weights.csv', '')
                    if file_name in name_mapping:
                        display_name = name_mapping[file_name]
                        weights_df = pd.read_csv(os.path.join(weights_dir, file), index_col=0)
                        weights_df.index = pd.to_datetime(weights_df.index)
                        portfolio_weights[display_name] = weights_df
                        logger.info(f"Loaded saved weights for {display_name}")
        
        if not portfolio_weights:
            logger.error("No portfolio weights loaded")
            return
        
        # 결과 디렉토리 설정 (window_size와 prediction_window 반영)
        result_dir = os.path.join(
            paths['results'],
            f'Result_{selections["data_size"]}_{selections["window_size"]}D{selections["prediction_window"]}P'
        )
        os.makedirs(os.path.join(result_dir, paths['figures']), exist_ok=True)
        logger.info(f"Created figures directory: {os.path.join(result_dir, paths['figures'])}")
        
        # 저장된 가중치 데이터 로드
        weights_dir = os.path.join(result_dir, paths['weights'])
        if not os.path.exists(weights_dir):
            logger.error(f"Weights directory not found: {weights_dir}")
            return
        
        portfolio_weights = {}
        
        # 파일명 -> 표시명 매핑
        name_mapping = portfolio_config['names']
        
        # 가중치 파일 로드
        weight_files = os.listdir(weights_dir)
        logger.info(f"Found {len(weight_files)} weight files in {weights_dir}")
        
        for file in weight_files:
            if file.endswith('_weights.csv'):
                file_name = file.replace('_weights.csv', '')
                if file_name in name_mapping:
                    display_name = name_mapping[file_name]
                    weights_df = pd.read_csv(os.path.join(weights_dir, file), index_col=0)
                    weights_df.index = pd.to_datetime(weights_df.index)
                    portfolio_weights[display_name] = weights_df
                    logger.info(f"Loaded weights for {display_name}")
        
        if not portfolio_weights:
            logger.error("No portfolio weights loaded")
            return
        
        # 시각화 옵션 선택
        viz_questions = [
            inquirer.List('viz_type',
                         message="Select visualization type",
                         choices=['Portfolio Comparison', 'Weight Distribution', 'Both'],
                         default='Portfolio Comparison'),
        ]
        
        if 'Portfolio Comparison' in viz_questions[0].choices:
            available_portfolios = sorted(portfolio_weights.keys())
            logger.info(f"Available portfolios: {available_portfolios}")
            viz_questions.append(
                inquirer.Checkbox('portfolios',
                                message="Select portfolios to compare",
                                choices=available_portfolios,
                                default=portfolio_config['default_selection'])
            )
        
        viz_selections = inquirer.prompt(viz_questions)
        if not viz_selections:
            logger.error("No visualization selections made")
            return
        
        logger.info(f"Selected visualization type: {viz_selections['viz_type']}")
        if 'portfolios' in viz_selections:
            logger.info(f"Selected portfolios: {viz_selections['portfolios']}")
        
        # 시각화 실행
        visualizer = PortfolioVisualizer()
        
        if viz_selections['viz_type'] in ['Portfolio Comparison', 'Both']:
            # Out of Sample 시작 시점
            investment_start = pd.to_datetime('2018-02-01')
            logger.info(f"Investment start date: {investment_start}")
            
            # 벤치마크 데이터 로드
            try:
                benchmark_returns = data_loader.load_benchmark()
                benchmark_returns = benchmark_returns[benchmark_returns.index >= investment_start]
                logger.info("Benchmark data loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load benchmark data: {str(e)}")
                benchmark_returns = None
            
            # 포트폴리오 수익률 계산
            portfolio_returns = {}
            metrics_results = {}
            metrics = PerformanceMetrics()
            
            for portfolio_name, weights in portfolio_weights.items():
                if portfolio_name in viz_selections.get('portfolios', []):
                    try:
                        returns_oos = returns_df[returns_df.index >= investment_start].copy()
                        portfolio_weights_oos = weights.reindex(returns_oos.index).ffill()
                        
                        # 일별 포트폴리오 수익률 계산
                        returns_series = (returns_oos * portfolio_weights_oos).sum(axis=1)
                        returns_series.name = portfolio_name
                        portfolio_returns[portfolio_name] = returns_series
                        
                        # 성과지표 계산 (벤치마크 추가)
                        metrics_results[portfolio_name] = metrics.calculate_portfolio_metrics(
                            returns=returns_series,
                            weights=portfolio_weights_oos,
                            benchmark_returns=benchmark_returns,  # 벤치마크 추가
                            result_dir=result_dir
                        )
                        
                        logger.info(f"Calculated returns and metrics for {portfolio_name}")
                        
                    except Exception as e:
                        logger.error(f"Error calculating returns for {portfolio_name}: {str(e)}")
                        continue
            
            if not portfolio_returns:
                logger.error("No portfolio returns calculated")
                return
            
            portfolio_returns_df = pd.DataFrame(portfolio_returns)
            metrics_df = pd.DataFrame(metrics_results).T
            
            # 소수점 4자리로 포맷
            metrics_df = metrics_df.round(4)
            
            # 성과지표 저장 (CSV, LaTeX)
            metrics_dir = os.path.join(result_dir, 'metrics')
            os.makedirs(metrics_dir, exist_ok=True)
            
            metrics_df.to_csv(os.path.join(metrics_dir, 'all_portfolios_metrics.csv'))
            metrics_df.to_latex(os.path.join(metrics_dir, 'all_portfolios_metrics.tex'))
            
            # 수익률 그래프 저장
            try:
                visualizer.plot_portfolio_comparison(
                    returns_dict=portfolio_returns_df,
                    weights_dict=portfolio_weights,
                    up_prob=up_prob,
                    title="Portfolio Performance Comparison",
                    result_dir=result_dir,
                    investment_start=investment_start,
                    rebalancing_freq='ME',
                    include_costs=False,
                    commission_rate=0.0003,
                    selected_portfolios=viz_selections.get('portfolios')
                )
                logger.info("Successfully created portfolio comparison plot")
                
                # 전략 비교 그래프 저장
                visualizer.plot_strategy_comparison(
                    metrics_df=metrics_df,
                    result_dir=result_dir,
                    title="Strategy Performance Comparison"
                )
                logger.info("Successfully created strategy comparison plot")
                
            except Exception as e:
                logger.error(f"Error creating visualization: {str(e)}", exc_info=True)
        
        if viz_selections['viz_type'] in ['Weight Distribution', 'Both']:
            logger.info("Creating weight distribution plots...")
            
            for portfolio_name, weights in portfolio_weights.items():
                if portfolio_name in viz_selections.get('portfolios', []):
                    try:
                        # 파일명에 사용할 수 없는 문자 제거
                        safe_name = "".join(x for x in portfolio_name if x.isalnum() or x in ['-', '_'])
                        
                        # 가중치 분포 플롯 생성
                        visualizer.plot_weights(
                            weights=weights,
                            save_path=os.path.join(result_dir, 'figures', f'{safe_name}_weights_plot.png'),
                            title=f"{portfolio_name} Portfolio Weights"
                        )
                        logger.info(f"Created weight distribution plot for {portfolio_name}")
                        
                    except Exception as e:
                        logger.error(f"Error creating weight plots for {portfolio_name}: {str(e)}")
                        continue

            logger.info("Visualization process completed")
        
        # 상위 수익률 종목 비중 시각화
        for portfolio_name, weights in portfolio_weights.items():
            if portfolio_name in viz_selections.get('portfolios', []):
                try:
                    safe_name = "".join(x for x in portfolio_name if x.isalnum() or x in ['-', '_'])
                    visualizer.plot_top_returns_weights(
                        returns=returns_df,
                        weights=weights,
                        investment_start=investment_start,
                        top_n=100,
                        save_path=os.path.join(result_dir, 'figures', f'{safe_name}_top_returns_weights.png'),
                        title=f'{portfolio_name} - Weight in Top 100 Return Stocks'
                    )
                    logger.info(f"Created top returns weights plot for {portfolio_name}")
                except Exception as e:
                    logger.error(f"Error creating top returns weights plot for {portfolio_name}: {str(e)}")
                    continue
        
        # 상위 수익률 종목 비중 비교 그래프
        try:
            visualizer.plot_top_returns_weights_comparison(
                returns=returns_df,
                portfolio_weights=portfolio_weights,
                investment_start=investment_start,
                top_n=100,
                save_path=os.path.join(result_dir, 'figures', 'top_returns_weights_comparison.png'),
                title='Portfolio Weight Comparison in Top 100 Return Stocks'
            )
            logger.info("Created top returns weights comparison plot")
        except Exception as e:
            logger.error(f"Error creating top returns weights comparison plot: {str(e)}")
        
        return
    
    # Full Process 실행
    if not selections['models']:
        print("At least one model must be selected")
        return
    
    # 선택 사항 출력
    logger.info(f"Selected mode: {selections['mode']}")
    logger.info(f"Selected data size: {selections['data_size']}")
    logger.info(f"Selected models: {selections['models']}")
    
    # 백테스트 실행
    backtest = PortfolioBacktest(
        base_folder=base_folder,
        data_size=selections['data_size'],
        selections=selections,
        train_date='2017-12-31',
        end_date='2024-07-05'
    )
    backtest.run()

if __name__ == "__main__":
    main()