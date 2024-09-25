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
        self.config = config
        self.len_pred = config['PRED_LEN']
        self.n_stock = config['N_FEAT']
        logger.info(f"Configuration: {self.config}")

    def _load_test_data(self) -> None:
        logger.info("Loading test data...")

        with open("data/dataset.pkl", "rb") as f:
            _, _, _, _, _, test_y_raw = pickle.load(f)

        with open("data/date.pkl", "rb") as f:
            date_info = pickle.load(f)

        self.test_y_raw = test_y_raw
        self.test_date = date_info['test']

    def _load_weights(self, model_identifiers: list) -> dict:
        weights_dict = {}
        for identifier in model_identifiers:
            model_result_dir = os.path.join(self.config['RESULT_DIR'], identifier)
            weights_path = os.path.join(model_result_dir, f"weights_{identifier}.npy")
            if os.path.exists(weights_path):
                weights = np.load(weights_path)
                weights_dict[identifier] = weights
                logger.info(f"Loaded weights for model: {identifier}")
            else:
                logger.warning(f"Weights not found for model: {identifier} at path: {weights_path}")

        return weights_dict

    def backtest(self, model_identifiers: list, visualize: bool = True) -> None:
        self._load_test_data()
        weights_dict = self._load_weights(model_identifiers)

        performance_dict = {}
        weights_over_time = {}
        ewp_performance = [10000]
        ewp_weights = np.ones(self.n_stock) / self.n_stock

        # Calculate EWP performance
        for i in range(0, len(self.test_y_raw), self.len_pred):
            m_rtn = np.sum(self.test_y_raw[i], axis=0)
            ewp_performance.append(ewp_performance[-1] * np.exp(np.dot(ewp_weights, m_rtn)))

        # Exclude the initial investment for alignment
        ewp_slice_end = len(self.test_date) // self.len_pred + 1
        performance_dict['EWP'] = ewp_performance[1:ewp_slice_end]

        # Calculate performance for each model
        for identifier, weights in weights_dict.items():
            model_performance = [10000]
            model_weights_over_time = []
            weights_index = 0
            for i in range(0, len(self.test_y_raw), self.len_pred):
                m_rtn = np.sum(self.test_y_raw[i], axis=0)
                portfolio_weights = weights[weights_index]
                
                model_performance.append(model_performance[-1] * np.exp(np.dot(portfolio_weights, m_rtn)))
                model_weights_over_time.append(portfolio_weights)
                
                weights_index += 1
            
            # Exclude the initial investment for alignment
            model_slice_end = len(self.test_date) // self.len_pred + 1
            performance_dict[identifier] = model_performance[1:model_slice_end]
            weights_over_time[identifier] = np.array(model_weights_over_time)

        # S&P 500 Benchmark Integration
        index_sp = pd.read_csv("data/snp500_index.csv", index_col="Date")
        index_sp.index = pd.to_datetime(index_sp.index)
        index_sp = index_sp[self.test_date[0]:]
        performance_dict['SPY'] = index_sp['Adj Close'].reindex(pd.to_datetime(self.test_date[::self.len_pred]), method='ffill') * (10000 / index_sp['Adj Close'].iloc[0])

        # Create performance DataFrame
        date_index = pd.to_datetime(self.test_date[::self.len_pred][:len(performance_dict['EWP'])])
        performance_df = pd.DataFrame(performance_dict, index=date_index)
        performance_df.to_csv(os.path.join(self.config['RESULT_DIR'], "combined_performance.csv"))

        # Visualization
        if visualize:
            visualize_backtest(performance_df, self.config)
            visualize_drawdown(performance_df, self.config)
            visualize_returns_distribution(performance_df, self.config)
            visualize_weights_over_time(weights_over_time, date_index, self.config)