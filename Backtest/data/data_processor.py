"""
데이터 전처리를 위한 모듈입니다.
주식 선택 및 포트폴리오 구성을 위한 데이터 처리 기능을 포함합니다.
"""

import pandas as pd
from typing import Dict, List, Tuple, Set
import logging
from datetime import datetime
import numpy as np

class DataProcessor:
    """
    데이터 전처리를 위한 클래스입니다.
    """
    
    def __init__(self):
        """
        DataProcessor 초기화
        """
        self.logger = logging.getLogger(__name__)

    def select_stocks(self, 
                     data: pd.DataFrame, 
                     n_stocks: int, 
                     top: bool = True) -> pd.DataFrame:
        """
        주어진 기준에 따라 주식을 선택합니다.
        
        Args:
            data (pd.DataFrame): 앙상블 결과 데이터
            n_stocks (int): 선택할 주식 수
            top (bool): True면 상위 주식, False면 하위 주식 선택
            
        Returns:
            pd.DataFrame: 선택된 주식 데이터
        """
        return data.sort_values(
            ['investment_date', 'up_prob'], 
            ascending=[True, not top]
        ).groupby('investment_date').head(n_stocks)

    def create_stock_dict(self, 
                         selected_stocks: pd.DataFrame) -> Dict[datetime, List[str]]:
        """
        선택된 주식들을 날짜별 딕셔너리로 변환합니다.
        
        Args:
            selected_stocks (pd.DataFrame): 선택된 주식 데이터
            
        Returns:
            Dict[datetime, List[str]]: 날짜별 주식 목록
        """
        return {
            date: selected_stocks[
                selected_stocks['investment_date'] == date
            ]['StockID'].tolist()
            for date in selected_stocks['investment_date'].unique()
        }

    def get_valid_dates(self, 
                    date: datetime, 
                    date_index: pd.DatetimeIndex) -> Tuple[datetime, datetime]:
        """
        주어진 날짜에 대한 유효한 시작일과 종료일을 반환합니다.
        
        Args:
            date (datetime): 기준 날짜
            date_index (pd.DatetimeIndex): 유효한 거래일 인덱스
            
        Returns:
            Tuple[datetime, datetime]: 유효한 시작일과 종료일
        """
        future_dates = date_index[date_index > pd.Timestamp(date)]
        past_dates = date_index[date_index < pd.Timestamp(date)]
        
        next_date = future_dates[0] if len(future_dates) > 0 else None
        prev_date = past_dates[-1] if len(past_dates) > 0 else None
        
        return next_date, prev_date

    def process_portfolio_stocks(self, 
                               up_prob_df: pd.DataFrame,
                               n_stocks: int = 100,
                               select_top: bool = True) -> Dict[datetime, List[str]]:
        """
        상승확률 기반 포트폴리오 구성을 위한 주식들을 처리합니다.
        
        Args:
            up_prob_df (pd.DataFrame): 상승확률 데이터
            n_stocks (int): 선택할 주식 수
            select_top (bool): True면 상위 주식, False면 하위 주식 선택
            
        Returns:
            Dict[datetime, List[str]]: 날짜별 선택된 주식 목록
        """
        selected_stocks = {}
        
        for date in up_prob_df.index:
            probs = up_prob_df.loc[date].sort_values(ascending=not select_top)
            selected_stocks[date] = probs.head(n_stocks).index.tolist()
            
        return selected_stocks

    def calculate_portfolio_returns(self,
                                  returns: pd.DataFrame,
                                  weights: pd.DataFrame,
                                  transaction_cost: float = 0.00015) -> Tuple[pd.Series, pd.Series]:
        """
        포트폴리오 수익률을 계산합니다.
        
        Args:
            returns (pd.DataFrame): 일간 수익률 데이터
            weights (pd.DataFrame): 포트폴리오 가중치
            transaction_cost (float): 거래 비용
            
        Returns:
            Tuple[pd.Series, pd.Series]: (누적 수익률, 순수익률)
        """
        # 가중치 검증
        if not np.allclose(weights.sum(axis=1), 1.0, rtol=1e-3):
            self.logger.warning("Portfolio weights do not sum to 1, normalizing...")
            weights = weights.div(weights.sum(axis=1), axis=0)
        
        # 음수 가중치 체크
        if (weights < 0).any().any():
            self.logger.error("Negative weights detected")
            raise ValueError("Negative weights are not allowed")
        
        # 1. 먼저 가중치의 시작일부터 끝일까지의 수익률만 사용
        start_date = weights.index[0]
        end_date = weights.index[-1]
        returns = returns.loc[start_date:end_date]
        
        # 2. 일별 포트폴리오 수익률 계산
        daily_returns = pd.Series(index=returns.index, dtype=float)
        
        # 3. 각 리밸런싱 기간별로 수익률 계산
        for i in range(len(weights.index)-1):
            current_date = weights.index[i]
            next_date = weights.index[i+1]
            
            # 현재 리밸런싱 기간의 가중치
            current_weights = weights.loc[current_date]
            
            # 해당 기간 동안의 수익률
            period_returns = returns.loc[current_date:next_date]
            
            # 일별 포트폴리오 수익률 계산
            for date in period_returns.index:
                daily_returns[date] = (period_returns.loc[date] * current_weights).sum()
        
        # 마지막 기간 처리
        last_weights = weights.iloc[-1]
        last_period_returns = returns.loc[weights.index[-1]:]
        for date in last_period_returns.index:
            daily_returns[date] = (last_period_returns.loc[date] * last_weights).sum()
        
        # 4. 거래비용 계산
        turnover = pd.Series(0.0, index=daily_returns.index)
        for i in range(1, len(weights)):
            turnover.loc[weights.index[i]] = np.abs(weights.iloc[i] - weights.iloc[i-1]).sum()
        
        transaction_costs = turnover * transaction_cost
        
        # 5. 순수익률 계산
        net_daily_returns = daily_returns - transaction_costs
        
        # 6. 누적수익률 계산
        cum_returns = (1 + daily_returns).cumprod()
        cum_net_returns = (1 + net_daily_returns).cumprod()
        
        # 디버깅 로그 추가
        self.logger.debug(f"Daily returns range: {daily_returns.index[0]} to {daily_returns.index[-1]}")
        self.logger.debug(f"Number of valid returns: {daily_returns.count()}")
        self.logger.debug(f"Number of NaN returns: {daily_returns.isna().sum()}")
        
        return cum_returns, cum_net_returns