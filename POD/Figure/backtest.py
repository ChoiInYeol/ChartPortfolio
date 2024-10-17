# backtest.py
import os
import numpy as np
import pandas as pd
import pickle
import logging
from typing import Dict, Any, List
from .visualize import (
    visualize_backtest,
    visualize_returns_distribution,
    visualize_drawdown,
    visualize_weights_over_time,
)
import h5py

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

class Backtester:
    def __init__(self, config: Dict[str, Any]):
        """
        백테스터 클래스를 초기화합니다.

        Args:
            config (Dict[str, Any]): 설정 정보를 담은 딕셔너리

        Returns:
            None
        """
        self.config = config
        self.len_pred = config['PRED_LEN']
        self.n_stock = config['N_FEAT']
        logger.info(f"설정: {self.config}")

    def _load_test_data(self) -> None:
        """
        테스트 데이터를 로드합니다.

        Returns:
            None
        """
        logger.info("테스트 데이터 로딩 중...")

        with h5py.File("data/test_dataset.h5", "r") as f:
            self.test_y_raw = f['y'][:, :, :50]  # 상위 50개 종목만 선택
        
        with open("data/test_times.pkl", "rb") as f:
            self.test_date = pickle.load(f)

        print(self.test_y_raw.shape)

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
        """
        백테스트를 수행합니다.

        Args:
            model_identifiers (list): 모델 식별자 리스트
            visualize (bool): 시각화 여부

        Returns:
            None
        """
        self._load_test_data()
        weights_dict = self._load_weights(model_identifiers)

        performance_dict = {}
        weights_over_time = {}
        initial_investment = 10000
        ewp_performance = [initial_investment]
        ewp_weights = np.ones(50) / 50  # 50개 종목에 대한 동일 가중치

        # EWP 성과 계산
        for i in range(0, len(self.test_y_raw), self.len_pred):
            m_rtn = np.sum(self.test_y_raw[i], axis=0)
            ewp_performance.append(ewp_performance[-1] * np.exp(np.dot(ewp_weights, m_rtn)))

        # 초기 투자금 제외
        ewp_slice_end = len(self.test_date) // self.len_pred + 1
        performance_dict['EWP'] = ewp_performance[1:ewp_slice_end]

        # Calculate performance for each model
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
            
            # Exclude the initial investment for alignment
            model_slice_end = len(self.test_date) // self.len_pred + 1
            performance_dict[identifier] = model_performance[1:model_slice_end]
            weights_over_time[identifier] = np.array(model_weights_over_time)

        # S&P 500 벤치마크 통합
        index_sp = pd.read_csv("data/snp500_index.csv", index_col="Date", parse_dates=True)
        
        # 2018년부터 시작
        start_date = pd.to_datetime("2018-01-01")
        
        print("start_date : ", start_date)
        print(index_sp.index)
        
        # 시작 날짜 이후의 데이터만 선택
        index_sp = index_sp[index_sp.index >= start_date]
        
        if index_sp.empty:
            logger.warning("선택된 시작 날짜 이후의 S&P 500 데이터가 없습니다.")
            performance_dict['SPY'] = pd.Series(dtype=float)
        else:
            # 성능 계산을 위한 리샘플링
            performance_dict['SPY'] = index_sp['Adj Close'].resample(f'{self.len_pred}D').last().ffill() * (initial_investment / index_sp['Adj Close'].iloc[0])

        # 성능 DataFrame 생성
        # self.test_date를 1차원 배열로 변환
        flat_test_dates = [date for sublist in self.test_date for date in sublist]
        date_index = pd.to_datetime(flat_test_dates[::self.len_pred][:len(performance_dict['EWP'])])
        performance_df = pd.DataFrame(performance_dict, index=date_index)
        performance_df.to_csv(os.path.join(self.config['RESULT_DIR'], "combined_performance.csv"))

        # Visualization
        if visualize:
            visualize_backtest(performance_df, self.config)
            visualize_drawdown(performance_df, self.config)
            visualize_returns_distribution(performance_df, self.config)
            visualize_weights_over_time(weights_over_time, date_index, self.config)
