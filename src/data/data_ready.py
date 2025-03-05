"""주식 데이터 전처리 및 준비 모듈

이 모듈은 주식 데이터의 다운로드, 필터링, 전처리를 담당합니다.
주요 기능:
1. S&P 500 지수 데이터 다운로드
2. 이상치 제거 및 필터링
3. 수익률 데이터프레임 생성
4. S&P 500 구성종목 수익률 데이터프레임 생성

Classes:
    StockDataProcessor: 주식 데이터 처리를 위한 메인 클래스

Dependencies:
    - pandas
    - numpy
    - scipy
    - FinanceDataReader
    - logging
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
from scipy import stats


class StockDataProcessor:
    """주식 데이터 처리를 위한 클래스
    
    이 클래스는 주식 데이터의 다운로드, 필터링, 전처리를 위한 메서드들을 제공합니다.
    
    Attributes:
        log_dir (str): 로그 파일 저장 디렉토리
        data_dir (Path): 데이터 파일 저장 디렉토리
        logger (logging.Logger): 로깅을 위한 logger 객체
    """

    def __init__(self, log_dir: str = 'logs', data_dir: Optional[Path] = None):
        """StockDataProcessor 초기화
        
        Args:
            log_dir (str): 로그 파일 저장 디렉토리
            data_dir (Optional[Path]): 데이터 파일 저장 디렉토리
        """
        self.log_dir = log_dir
        self.data_dir = data_dir or Path(__file__).parent
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """로깅 설정을 초기화합니다.
        
        Returns:
            logging.Logger: 설정된 logger 객체
        """
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.log_dir, f'stock_filter_{timestamp}.log')
        
        logger = logging.getLogger('StockDataProcessor')
        logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        stream_handler = logging.StreamHandler()
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        
        logger.info("Logging initialized")
        return logger

    def download_market_data(self, index_name: str = 'S&P500') -> pd.DataFrame:
        """시장 지수 데이터를 다운로드합니다.
        
        Args:
            index_name (str): 다운로드할 지수 이름
        
        Returns:
            pd.DataFrame: 다운로드된 지수 데이터
            
        Raises:
            Exception: 데이터 다운로드 실패시 발생
        """
        try:
            market_data = fdr.DataReader(index_name)
            market_data.index.name = 'Date'
            output_path = self.data_dir / f"{index_name.lower().replace('&', '')}_index.csv"
            market_data.to_csv(output_path)
            self.logger.info(f"{index_name} data downloaded and saved to {output_path}")
            return market_data
        except Exception as e:
            self.logger.error(f"{index_name} data download failed: {e}")
            raise

    def filter_stocks(
        self,
        input_file: Union[str, Path],
        output_file: Union[str, Path],
        min_trading_days: int = 1000,
        start_date: str = '2001-01-01',
        end_date: str = '2024-08-01',
        confidence_level: float = 0.95
    ) -> pd.DataFrame:
        """주식 데이터에서 이상치를 제거하고 필터링합니다.
        
        Args:
            input_file: 입력 CSV 파일 경로
            output_file: 출력 CSV 파일 경로
            min_trading_days: 최소 거래일수
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)
            confidence_level: 이상치 제거를 위한 신뢰수준
            
        Returns:
            pd.DataFrame: 필터링된 주식 데이터
            
        Raises:
            Exception: 데이터 처리 중 오류 발생시
        """
        self.logger.info("Starting stock data filtering process")
        
        try:
            # 데이터 로드
            df = pd.read_csv(input_file)
            # 날짜 형식을 자동으로 인식하도록 format 매개변수 제거
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # 결측 날짜 제거
            df = df.dropna(subset=['date'])
            
            # 날짜 범위 필터링
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_date)
            df = df[(df['date'] >= start_date_dt) & (df['date'] <= end_date_dt)]
            
            # 거래일수 기준 필터링
            trading_days = df.groupby('PERMNO').size()
            valid_stocks = trading_days[trading_days >= min_trading_days].index
            df = df[df['PERMNO'].isin(valid_stocks)]
            
            self.logger.info(f"Stocks with sufficient trading days: {len(valid_stocks)}")
            
            # 통계량 계산
            stock_stats = self._calculate_stock_statistics(df, valid_stocks)
            
            # t-통계량 기반 이상치 제거
            valid_stocks = self._remove_outliers(
                stock_stats, 
                confidence_level=confidence_level
            )
            
            # 최종 데이터셋 생성
            filtered_df = df[df['PERMNO'].isin(valid_stocks)].copy()
            filtered_df.sort_values(['PERMNO', 'date'], inplace=True)
            
            # 결과 저장
            filtered_df.to_csv(output_file, index=False)
            
            self.logger.info(f"Filtering complete. Remaining stocks: {len(valid_stocks)}")
            self.logger.info(f"Filtered data saved to {output_file}")
            
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"Error in filter_stocks: {str(e)}")
            raise

    def _calculate_stock_statistics(
        self, 
        df: pd.DataFrame, 
        stock_ids: List
    ) -> pd.DataFrame:
        """각 주식의 통계량을 계산합니다.
        
        Args:
            df: 주식 데이터
            stock_ids: 주식 ID 리스트
            
        Returns:
            pd.DataFrame: 계산된 통계량
        """
        stats_list = []
        for stock_id in stock_ids:
            stock_data = df[df['PERMNO'] == stock_id]
            
            stats_dict = {
                'PERMNO': stock_id,
                'mean_ret': stock_data['RET'].mean(),
                'std_ret': stock_data['RET'].std(),
                'skew_ret': stats.skew(stock_data['RET']),
                'kurt_ret': stats.kurtosis(stock_data['RET']),
                'trading_days': len(stock_data)
            }
            stats_list.append(stats_dict)
            
        return pd.DataFrame(stats_list)

    def _remove_outliers(
        self, 
        stats_df: pd.DataFrame, 
        confidence_level: float = 0.95
    ) -> set:
        """통계량 기반으로 이상치를 제거합니다.
        
        Args:
            stats_df: 주식별 통계량 데이터프레임
            confidence_level: 신뢰수준
            
        Returns:
            set: 이상치가 제거된 주식 ID 집합
        """
        z_critical = stats.norm.ppf(confidence_level)
        metrics = ['mean_ret', 'std_ret', 'skew_ret', 'kurt_ret']
        
        valid_stocks = set(stats_df['PERMNO'])
        for metric in metrics:
            z_scores = np.abs(stats.zscore(stats_df[metric]))
            extreme_stocks = stats_df[z_scores > z_critical]['PERMNO']
            valid_stocks -= set(extreme_stocks)
            self.logger.info(f"Removed {len(extreme_stocks)} stocks based on {metric}")
        
        return valid_stocks

    def create_return_df(
        self, 
        filtered_stocks_path: Union[str, Path],
        symbol_permno_path: Union[str, Path],
        output_file: Optional[Union[str, Path]] = None,
        sp500_only: bool = False
    ) -> pd.DataFrame:
        """수익률 데이터프레임을 생성합니다.
        
        Args:
            filtered_stocks_path: 필터링된 주식 데이터 파일 경로
            symbol_permno_path: Symbol-PERMNO 매핑 파일 경로
            output_file: 출력 파일 경로 (선택사항)
            sp500_only: S&P 500 구성종목만 포함할지 여부
            
        Returns:
            pd.DataFrame: 생성된 수익률 데이터프레임
        """
        # Symbol-PERMNO 매핑 로드
        symbol_permno = pd.read_csv(symbol_permno_path)
        
        # 필터링된 주식 데이터 로드
        stocks_df = pd.read_csv(filtered_stocks_path, parse_dates=['date'])
        
        if sp500_only:
            # SP500 구성종목 필터링
            sp500_path = self.data_dir / 'sp500_20180101.csv'
            sp500_tickers = pd.read_csv(sp500_path)['Symbol'].tolist()
            sp500_permnos = symbol_permno[
                symbol_permno['Symbol'].isin(sp500_tickers)
            ]['PERMNO'].tolist()
            stocks_df = stocks_df[stocks_df['PERMNO'].isin(sp500_permnos)]
        
        # PERMNO를 Symbol로 변환
        permno_to_symbol = dict(zip(symbol_permno['PERMNO'], symbol_permno['Symbol']))
        stocks_df.loc[:, 'Symbol'] = stocks_df['PERMNO'].map(permno_to_symbol)
        
        # 수익률 데이터프레임 생성
        return_df = stocks_df.pivot(
            index='date',
            columns='Symbol',
            values='RET'
        )
        
        # 결과 저장
        if output_file:
            return_df.to_csv(output_file)
            self.logger.info(f"Return data saved to {output_file}")
        
        self.logger.info(f"Return data created: {return_df.shape}")
        self.logger.info(f"Period: {return_df.index.min()} ~ {return_df.index.max()}")
        self.logger.info(f"Number of stocks: {len(return_df.columns)}")
        
        return return_df 