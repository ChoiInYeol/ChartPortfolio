"""주식 데이터 다운로드 모듈

이 모듈은 주식 시장 데이터의 다운로드와 초기 처리를 담당합니다.
주요 기능:
1. NASDAQ, NYSE 상장 종목 정보 수집
2. 개별 종목 가격 데이터 다운로드
3. S&P 500 구성종목 데이터 처리
4. 통합 데이터셋 생성

Classes:
    StockDataDownloader: 주식 데이터 다운로드를 위한 메인 클래스

Dependencies:
    - pandas
    - numpy
    - FinanceDataReader
    - concurrent.futures
    - tqdm
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
from tqdm import tqdm


class StockDataDownloader:
    """주식 데이터 다운로드 및 처리를 위한 클래스
    
    이 클래스는 주식 데이터의 다운로드, 필터링, 초기 처리를 위한 메서드들을 제공합니다.
    
    Attributes:
        data_dir (Path): 데이터 저장 디렉토리
        logger (logging.Logger): 로깅을 위한 logger 객체
        min_history_days (int): 필터링을 위한 최소 거래일수
    """

    def __init__(
        self, 
        data_dir: Optional[Path] = None,
        min_history_days: int = 5000
    ):
        """StockDataDownloader 초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리 (기본값: '../../data/raw')
            min_history_days: 필터링을 위한 최소 거래일수
        """
        self.data_dir = data_dir or Path('../../data/raw')
        self.min_history_days = min_history_days
        self.logger = self._setup_logging()
        self._create_directories()

    def _setup_logging(self) -> logging.Logger:
        """로깅 설정을 초기화합니다."""
        logger = logging.getLogger('StockDataDownloader')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 파일 핸들러
        file_handler = logging.FileHandler('download.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

    def _create_directories(self) -> None:
        """필요한 디렉토리들을 생성합니다."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def get_exchange_symbols(self) -> pd.DataFrame:
        """나스닥과 뉴욕거래소의 종목 정보를 추출합니다.
        
        Returns:
            pd.DataFrame: 거래소 종목 정보
            
        Notes:
            이미 다운로드된 종목은 제외됩니다.
        """
        # 거래소별 종목 리스트 다운로드
        nasdaq = fdr.StockListing('NASDAQ')
        nyse = fdr.StockListing('NYSE')
        
        # 거래소 정보 추가
        nasdaq['Exchange'] = 'NASDAQ'
        nyse['Exchange'] = 'NYSE'
        
        # 데이터프레임 통합
        symbols = pd.concat([nasdaq, nyse])
        symbols = symbols[['Symbol', 'Name', 'IndustryCode', 'Industry', 'Exchange']]
        
        # 이미 다운로드된 종목 제외
        if os.path.exists('symbol_permno.csv'):
            downloaded = pd.read_csv('symbol_permno.csv')
            symbols = symbols[~symbols['Symbol'].isin(downloaded['Symbol'])]
            self.logger.info(f"Excluding {len(downloaded)} already downloaded symbols")
        
        return symbols

    def _process_single_stock(
        self, 
        symbol: str,
        exchange: str
    ) -> Optional[Tuple[str, pd.DataFrame]]:
        """단일 종목의 데이터를 다운로드하고 처리합니다.
        
        Args:
            symbol: 종목 코드
            exchange: 거래소 정보
            
        Returns:
            Optional[Tuple[str, pd.DataFrame]]: 처리된 종목 데이터
        """
        try:
            data = fdr.DataReader(symbol)
            
            if len(data) >= self.min_history_days:
                # 조정 계수 계산
                adj_factor = data['Adj Close'] / data['Close']
                
                # 가격 데이터 조정
                data['Open'] = data['Open'] * adj_factor
                data['High'] = data['High'] * adj_factor
                data['Low'] = data['Low'] * adj_factor
                data['Close'] = data['Adj Close']
                
                # 결측치 처리
                data = data.ffill()
                
                # 심볼 및 거래소 정보 추가
                data['Symbol'] = symbol
                data['Exchange'] = exchange
                data.index.name = 'date'
                
                # 파일 저장
                data.to_csv(self.data_dir / f'{symbol}.csv')
                self.logger.info(f"Downloaded and processed {symbol}")
                
                return symbol, data
                
            self.logger.info(
                f"Skipped {symbol} (insufficient data: {len(data)} days)"
            )
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {e}")
            return None

    def download_and_filter_stocks(
        self, 
        symbols: pd.DataFrame,
        max_workers: int = 8
    ) -> pd.DataFrame:
        """종목 데이터를 다운로드하고 필터링합니다.
        
        Args:
            symbols: 다운로드할 종목 정보
            max_workers: 동시 다운로드 작업자 수
            
        Returns:
            pd.DataFrame: 필터링된 종목 정보
        """
        if len(symbols) == 0:
            self.logger.info("No new symbols to download")
            return pd.DataFrame()
        
        filtered_symbols = []
        
        # 멀티스레딩으로 다운로드 및 필터링
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self._process_single_stock, 
                    row['Symbol'],
                    row['Exchange']
                )
                for _, row in symbols.iterrows()
            ]
            
            for future in tqdm(
                as_completed(futures), 
                total=len(symbols),
                desc="Downloading and filtering"
            ):
                result = future.result()
                if result is not None:
                    symbol, _ = result
                    filtered_symbols.append(
                        symbols[symbols['Symbol'] == symbol].iloc[0]
                    )
        
        filtered_df = pd.DataFrame(filtered_symbols)
        self._update_symbol_permno(filtered_df)
        
        self.logger.info(f"Added {len(filtered_df)} new symbols")
        return filtered_df

    def _update_symbol_permno(self, new_symbols: pd.DataFrame) -> None:
        """Symbol-PERMNO 매핑을 업데이트합니다.
        
        Args:
            new_symbols: 새로 추가된 종목 정보
        """
        if os.path.exists('symbol_permno.csv'):
            existing_symbols = pd.read_csv('symbol_permno.csv')
            next_permno = existing_symbols['PERMNO'].max() + 1
            
            new_mapping = pd.DataFrame({
                'Symbol': new_symbols['Symbol'],
                'PERMNO': range(next_permno, next_permno + len(new_symbols))
            })
            
            updated_symbols = pd.concat([existing_symbols, new_mapping])
            updated_symbols.to_csv('symbol_permno.csv', index=False)
        else:
            symbol_permno = pd.DataFrame({
                'Symbol': new_symbols['Symbol'],
                'PERMNO': range(1, len(new_symbols) + 1)
            })
            symbol_permno.to_csv('symbol_permno.csv', index=False)

    def create_final_dataset(self, symbols: pd.DataFrame) -> None:
        """최종 데이터셋을 생성합니다.
        
        Args:
            symbols: 필터링된 종목 정보
        """
        dfs = []
        
        for file in tqdm(
            list(self.data_dir.glob('*.csv')), 
            desc="Merging data"
        ):
            try:
                df = pd.read_csv(file)
                df['date'] = pd.to_datetime(df['date'])
                
                # 수익률 계산
                df.sort_values('date', inplace=True)
                df['Ret'] = df['Close'].pct_change()
                
                # 결측치 제거
                df.dropna(subset=['Ret'], inplace=True)
                
                # 종목 정보 추가
                symbol = file.stem
                symbol_info = symbols[symbols['Symbol'] == symbol].iloc[0]
                df['Exchange'] = symbol_info['Exchange']
                df['IndustryCode'] = symbol_info['IndustryCode']
                
                dfs.append(df)
            except Exception as e:
                self.logger.error(f"Error processing {file}: {e}")
        
        if dfs:
            final_data = self._process_final_dataset(pd.concat(dfs, ignore_index=True))
            final_data.to_csv('Data.csv', index=False)
            
            self.logger.info(
                f"Final dataset created with {len(symbols)} symbols"
            )

    def _process_final_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        """최종 데이터셋을 처리합니다.
        
        Args:
            data: 통합된 원본 데이터
            
        Returns:
            pd.DataFrame: 처리된 최종 데이터셋
        """
        # PERMNO 매핑
        symbol_permno = pd.read_csv('symbol_permno.csv')
        data = data.merge(symbol_permno, on='Symbol')
        
        # 날짜 형식 변경
        data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y%m%d')
        
        # 컬럼 이름 변경
        columns = {
            'date': 'date',
            'PERMNO': 'PERMNO',
            'Low': 'BIDLO',
            'High': 'ASKHI',
            'Close': 'PRC',
            'Volume': 'VOL',
            'Open': 'OPENPRC',
            'Ret': 'RET'
        }
        
        data = data.rename(columns=columns)
        data['SHROUT'] = np.nan  # Shares Outstanding 정보 없음
        
        # 최종 컬럼 선택 및 정렬
        final_columns = [
            'date', 'PERMNO', 'BIDLO', 'ASKHI', 'PRC', 
            'VOL', 'SHROUT', 'OPENPRC', 'RET'
        ]
        
        return (data[final_columns]
                .round(3)
                .sort_values(['PERMNO', 'date']))

    def check_sp500(self, symbols: pd.DataFrame) -> None:
        """S&P 500 종목을 확인하고 누락된 종목을 다운로드합니다.
        
        Args:
            symbols: 현재 보유 중인 종목 정보
        """
        try:
            # S&P 500 종목 리스트 로드
            sp500 = pd.read_csv('../meta/sp500_20180101.csv')
            sp500_symbols = set(sp500['Symbol'])
            
            # 현재 보유 종목 확인
            downloaded_symbols = set(symbols['Symbol'])
            
            # 이전 오류 기록 확인 및 처리
            error_records = self._load_error_records()
            error_symbols = set(error_records['Symbol']) if error_records is not None else set()
            
            # 누락된 종목 확인 및 처리
            missing_sp500 = sp500_symbols - downloaded_symbols - error_symbols
            
            if missing_sp500:
                self._process_missing_sp500(missing_sp500, error_records)
            else:
                self.logger.info("No missing S&P 500 stocks to download")
                
        except Exception as e:
            self.logger.error(f"Error checking S&P 500 stocks: {e}")

    def _load_error_records(self) -> Optional[pd.DataFrame]:
        """오류 기록을 로드합니다."""
        if os.path.exists('sp500_ticker.csv'):
            return pd.read_csv('sp500_ticker.csv')
        return None

    def _process_missing_sp500(
        self, 
        missing_symbols: set,
        error_records: Optional[pd.DataFrame]
    ) -> None:
        """누락된 S&P 500 종목을 처리합니다."""
        self.logger.info(f"Attempting to download {len(missing_symbols)} S&P 500 stocks")
        
        error_list = []
        for symbol in tqdm(missing_symbols, desc="Downloading missing S&P 500 stocks"):
            try:
                missing_df = pd.DataFrame({'Symbol': [symbol]})
                self.download_and_filter_stocks(missing_df)
            except Exception as e:
                error_list.append({
                    'Symbol': symbol,
                    'Error_Type': type(e).__name__,
                    'Error_Message': str(e),
                    'Timestamp': pd.Timestamp.now()
                })
        
        if error_list:
            self._update_error_records(error_list, error_records)
            self.logger.info(f"Errors occurred for {len(error_list)} stocks")

    def _update_error_records(
        self, 
        error_list: List[Dict],
        existing_records: Optional[pd.DataFrame]
    ) -> None:
        """오류 기록을 업데이트합니다."""
        new_errors = pd.DataFrame(error_list)
        if existing_records is not None:
            error_records = pd.concat([existing_records, new_errors], ignore_index=True)
        else:
            error_records = new_errors
        
        error_records.to_csv('sp500_ticker.csv', index=False) 