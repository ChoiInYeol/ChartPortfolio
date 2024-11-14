# backtest.py
import os
import numpy as np
import pandas as pd
import pickle
import logging
from typing import Dict, Any
from visualize import (
    visualize_backtest,
    visualize_returns_distribution,
    visualize_drawdown,
    visualize_weights_over_time,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

class Backtester:
    def __init__(self, config: Dict[str, Any]):
        """백테스터 초기화"""
        self.config = config
        self.len_pred = config['PRED_LEN']
        self.n_stock = config['N_STOCK']
        logger.info(f"Configuration: {self.config}")

    def _load_test_data(self) -> None:
        """테스트 데이터 로드"""
        logger.info("Loading test data...")

        with open("data/dataset.pkl", "rb") as f:
            _, _, _, _, test_x_raw, test_y_raw, \
            _, _, test_prob_raw, _, _, _ = pickle.load(f)

        self.test_y_raw = test_y_raw
        self.test_prob = test_prob_raw
        with open("data/date.pkl", "rb") as f:
            date_info = pickle.load(f)
        self.test_date = date_info['test']

    def _load_weights(self, model_identifiers: list) -> dict:
        """모델 가중치 로드"""
        weights_dict = {}
        for identifier in model_identifiers:
            weights_path = os.path.join(self.config['RESULT_DIR'], f"weights_{identifier}.npy")
            if os.path.exists(weights_path):
                weights = np.load(weights_path)
                weights_dict[identifier] = weights
                logger.info(f"Loaded weights for model: {identifier}")
            else:
                logger.warning(f"Weights not found for model: {identifier} at path: {weights_path}")
        return weights_dict

    def backtest(self, model_identifiers: list, visualize: bool = True) -> None:
        """백테스트 수행"""
        self._load_test_data()
        weights_dict = self._load_weights(model_identifiers)

        # 날짜 인덱스 생성
        test_dates = pd.DatetimeIndex(self.test_date)
        test_dates_monthly = test_dates[::self.len_pred]  # 20일 간격으로 선택

        # 성과 저장을 위한 딕셔너리 초기화
        performance_dict = {}
        weights_over_time = {}
        initial_investment = 10000

        # EWP (Equal Weight Portfolio) 계산
        ewp_performance = [initial_investment]
        ewp_weights = np.ones(self.n_stock) / self.n_stock

        for i in range(0, len(self.test_y_raw), self.len_pred):
            m_rtn = np.sum(self.test_y_raw[i], axis=0)
            ewp_performance.append(ewp_performance[-1] * np.exp(np.dot(ewp_weights, m_rtn)))

        performance_dict['EWP'] = ewp_performance[1:]

        # 각 모델의 성과 계산
        for identifier, weights in weights_dict.items():
            model_performance = [initial_investment]
            model_weights_over_time = []
            weights_index = 0

            for i in range(0, len(self.test_y_raw), self.len_pred):
                m_rtn = np.sum(self.test_y_raw[i], axis=0)
                portfolio_weights = weights[weights_index]
                
                model_performance.append(model_performance[-1] * np.exp(np.dot(portfolio_weights, m_rtn)))
                model_weights_over_time.append(portfolio_weights)
                weights_index += 1

            performance_dict[identifier] = model_performance[1:]
            weights_over_time[identifier] = np.array(model_weights_over_time)

        # S&P 500 벤치마크 통합
        try:
            index_sp = pd.read_csv("data/snp500_index.csv", index_col="Date", parse_dates=True)
            index_sp = index_sp[test_dates[0]:test_dates[-1]]
            
            # 20일 간격으로 리샘플링
            spy_series = index_sp['Adj Close'].reindex(test_dates_monthly)
            spy_series = spy_series * (initial_investment / spy_series.iloc[0])
            performance_dict['SPY'] = spy_series.values
            
        except Exception as e:
            logger.error(f"Error processing S&P 500 data: {str(e)}")

        # 성과 DataFrame 생성
        performance_df = pd.DataFrame(performance_dict, index=test_dates_monthly[:len(performance_dict['EWP'])])
        performance_df.to_csv(os.path.join(self.config['RESULT_DIR'], "combined_performance.csv"))

        # 시각화
        if visualize:
            visualize_backtest(performance_df, self.config)
            visualize_drawdown(performance_df, self.config)
            visualize_returns_distribution(performance_df, self.config)
            visualize_weights_over_time(weights_over_time, performance_df.index, self.config)