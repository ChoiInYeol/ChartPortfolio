"""데이터 파이프라인 모듈

이 모듈은 주식 데이터의 다운로드부터 전처리까지의 전체 과정을 관리합니다.
주요 기능:
1. 주식 데이터 다운로드 및 필터링
2. 데이터 전처리 및 정제
3. 최종 분석용 데이터셋 생성

Classes:
    DataPipeline: 데이터 처리 파이프라인 메인 클래스
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Set
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
from dataclasses import dataclass
import shutil
import yfinance as yf

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.data_download import StockDataDownloader
from src.data.data_ready import StockDataProcessor


@dataclass
class PipelineStatus:
    """파이프라인 실행 상태를 저장하는 클래스"""
    need_stock_download: bool = False
    need_sp500_check: bool = False
    need_processing: bool = False
    missing_files: List[str] = None
    existing_files: List[str] = None
    total_symbols: int = 0
    downloaded_symbols: int = 0
    downloaded_files_count: int = 0  # 다운로드된 파일 수

    def __str__(self) -> str:
        """상태 정보를 문자열로 반환합니다."""
        status = [
            "\n=== Pipeline Status ===",
            f"Total Symbols: {self.total_symbols}",
            f"Downloaded Symbols: {self.downloaded_symbols}",
            f"Downloaded Files Count: {self.downloaded_files_count}",
            "\nRequired Steps:",
            f"- Stock Download: {'Yes' if self.need_stock_download else 'No'}",
            f"- SP500 Check: {'Yes' if self.need_sp500_check else 'No'}",
            f"- Data Processing: {'Yes' if self.need_processing else 'No'}",
            "\nMissing Files:",
            *[f"- {f}" for f in (self.missing_files or [])],
            "\nExisting Files:",
            *[f"- {f}" for f in (self.existing_files or [])]
        ]
        return "\n".join(status)


class DataPipeline:
    """데이터 처리 파이프라인 클래스
    
    전체 데이터 처리 과정을 관리하고 조율하는 클래스입니다.
    
    Attributes:
        base_dir (Path): 기본 데이터 디렉토리
        data_dir (Path): 처리된 데이터 저장 디렉토리
        logger (logging.Logger): 로깅을 위한 logger 객체
    """
    
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        min_history_days: int = 5000
    ):
        """DataPipeline 초기화
        
        Args:
            base_dir: 기본 데이터 디렉토리
            min_history_days: 필터링을 위한 최소 거래일수
        """
        self.base_dir = base_dir or Path(__file__).parent
        self.data_dir = self.base_dir
        self.logger = self._setup_logging()
        
        # 컴포넌트 초기화
        self.downloader = StockDataDownloader(
            data_dir=self.data_dir / 'raw',
            min_history_days=min_history_days
        )
        self.processor = StockDataProcessor(
            log_dir=str(self.data_dir / 'logs'),
            data_dir=self.data_dir
        )
        
        # 디렉토리 생성 및 구조 정리
        self._create_directories()
        self._organize_files()
        
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정을 초기화합니다."""
        logger = logging.getLogger('DataPipeline')
        logger.setLevel(logging.INFO)
        
        # 로그 디렉토리 생성
        log_dir = self.data_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 파일 핸들러
        file_handler = logging.FileHandler(log_dir / 'pipeline.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _create_directories(self) -> None:
        """필요한 디렉토리들을 생성합니다."""
        dirs = [
            self.data_dir / 'raw',           # 원본 데이터
            self.data_dir / 'processed',      # 처리된 데이터
            self.data_dir / 'logs'           # 로그 파일
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _organize_files(self) -> None:
        """파일 구조를 정리합니다."""
        try:
            # 로그 파일 이동
            log_files = ['download.log', 'pipeline.log']
            log_dir = self.data_dir / 'logs'
            
            for file in log_files:
                src = self.data_dir / file
                if src.exists():
                    dst = log_dir / file
                    if not dst.exists():
                        shutil.move(str(src), str(dst))
                        self.logger.info(f"Moved {file} to logs directory")
            
            # 메타데이터 파일 이동
            meta_files = ['sp500_ticker.csv', 'symbol_permno.csv']
            raw_dir = self.data_dir / 'raw'
            
            for file in meta_files:
                src = self.data_dir / file
                if src.exists():
                    dst = raw_dir / file
                    if not dst.exists():
                        shutil.move(str(src), str(dst))
                        self.logger.info(f"Moved {file} to raw directory")
                        
        except Exception as e:
            self.logger.error(f"Error organizing files: {str(e)}")
            
    def validate_pipeline(self, target_date: str) -> PipelineStatus:
        """파이프라인 실행 전 상태를 검증합니다.
        
        Args:
            target_date: 목표 기준일 (YYYY-MM-DD)
            
        Returns:
            PipelineStatus: 파이프라인 실행 상태
        """
        status = PipelineStatus()
        status.missing_files = []
        status.existing_files = []
        
        # 0. 기준일 처리
        target_date_str = target_date.replace('-', '')
        
        # 1. 필요한 파일 경로 설정
        required_files = {
            'processed_data': self.data_dir / 'processed' / 'Data.csv',
            'sp500_list': self.data_dir / 'processed' / f"sp500_{target_date_str}.csv",
            'symbol_permno': self.data_dir / 'processed' / 'symbol_permno.csv'
        }
        
        # 2. 파일 존재 여부 확인
        for name, path in required_files.items():
            if path.exists():
                status.existing_files.append(str(path))
            else:
                status.missing_files.append(str(path))
        
        # 3. 데이터 상태 확인
        raw_dir = self.data_dir / 'raw'
        if raw_dir.exists():
            # raw 디렉토리의 모든 CSV 파일 수 확인 (메타데이터 파일 제외)
            excluded_files = ['symbol_permno.csv', f'sp500_{target_date_str}.csv', 'sp500_ticker.csv']
            stock_files = [f for f in raw_dir.glob('*.csv') if f.name not in excluded_files]
            status.downloaded_files_count = len(stock_files)
            status.downloaded_symbols = status.downloaded_files_count
            
            # 2000개 이상의 종목 데이터가 있으면 다운로드 스킵
            if status.downloaded_files_count >= 2000:
                status.need_stock_download = False
                self.logger.info(f"Found {status.downloaded_files_count} stock files. Skipping download.")
            else:
                status.need_stock_download = True
                self.logger.info(f"Only {status.downloaded_files_count} stock files found. Download needed.")
        else:
            status.need_stock_download = True
            
        if required_files['processed_data'].exists():
            processed_data = pd.read_csv(required_files['processed_data'])
            status.downloaded_symbols = len(processed_data['PERMNO'].unique())
        
        # S&P 500 파일 우선 확인
        if required_files['sp500_list'].exists():
            sp500_data = pd.read_csv(required_files['sp500_list'])
            status.total_symbols = len(sp500_data)
            self.logger.info(f"Found {status.total_symbols} symbols in S&P 500 list")
        else:
            # sp500_ticker.csv 파일 확인 (대체 파일)
            alt_sp500_file = self.data_dir / 'processed' / 'sp500_ticker.csv'
            if alt_sp500_file.exists():
                sp500_data = pd.read_csv(alt_sp500_file)
                status.total_symbols = len(sp500_data)
                self.logger.info(f"Using alternative S&P 500 list with {status.total_symbols} symbols")
            else:
                self.logger.warning("No S&P 500 list found. Will create during pipeline run.")
                status.need_sp500_check = True
        
        # 4. 처리 필요 여부 확인
        status.need_processing = (
            status.need_stock_download or 
            not required_files['processed_data'].exists()
        )
        
        return status

    def run_pipeline(
        self,
        target_date: str = '2018-01-01',
        config: Optional[Dict[str, Any]] = None,
        force_rerun: bool = False,
        apply_tstat_filter: bool = True,
        confidence_level: float = 0.95,
        min_trading_days: int = 1000
    ) -> None:
        """데이터 파이프라인을 실행합니다.
        
        Args:
            target_date: 목표 기준일 (YYYY-MM-DD)
            config: 파이프라인 설정
            force_rerun: 강제 재실행 여부
            apply_tstat_filter: t-통계량 필터링 적용 여부
            confidence_level: 이상치 제거를 위한 신뢰수준
            min_trading_days: 최소 거래일수
        """
        try:
            # 타겟 날짜 형식 변환
            target_date_str = target_date.replace('-', '')
            
            # 파일 이동 및 정리
            self._organize_files()
            
            # 파이프라인 실행 상태 검증
            self.logger.info("Starting pipeline validation...")
            status = self.validate_pipeline(target_date)
            self.logger.info(str(status))
            
            if not force_rerun and not any([
                status.need_stock_download,
                status.need_sp500_check,
                status.need_processing
            ]):
                self.logger.info("All data is up to date. No processing needed.")
                return
            
            if force_rerun:
                self.logger.info("Force rerun enabled. Running all steps...")
            
            # 1. 주식 데이터 다운로드
            if status.need_stock_download or force_rerun:
                self.logger.info("Step 1: Downloading stock data...")
                symbols = self.downloader.get_exchange_symbols()
                filtered_symbols = self.downloader.download_and_filter_stocks(symbols)
            else:
                self.logger.info("Step 1: Stock data already exists, skipping download...")
                # symbol_permno.csv가 있으면 로드, 없으면 빈 데이터프레임 생성
                symbol_permno_path = self.data_dir / 'processed' / 'symbol_permno.csv'
                if symbol_permno_path.exists():
                    filtered_symbols = pd.read_csv(symbol_permno_path)
                else:
                    self.logger.warning("symbol_permno.csv not found, creating empty DataFrame")
                    filtered_symbols = pd.DataFrame(columns=['Symbol', 'PERMNO'])
            
            # 2. S&P 500 종목 확인
            # 먼저 target_date에 맞는 sp500_{target_date_str}.csv 파일이 있는지 확인
            sp500_file = self.data_dir / 'processed' / f"sp500_{target_date_str}.csv"
            if sp500_file.exists():
                self.logger.info(f"Using S&P 500 list for date {target_date}")
                sp500_df = pd.read_csv(sp500_file)
                # filtered_symbols에 S&P 500 정보 추가
                if not 'SP500' in filtered_symbols.columns:
                    filtered_symbols['SP500'] = False
                for symbol in sp500_df['Symbol'].values:
                    filtered_symbols.loc[filtered_symbols['Symbol'] == symbol, 'SP500'] = True
                # symbol_permno.csv 업데이트
                filtered_symbols.to_csv(self.data_dir / 'processed' / 'symbol_permno.csv', index=False)
                self.logger.info(f"Updated symbol_permno.csv with S&P 500 info")
            elif status.need_sp500_check or force_rerun:
                self.logger.info("Step 2: Checking S&P 500 stocks...")
                try:
                    self.downloader.check_sp500(filtered_symbols)
                    # S&P 500 파일을 processed 폴더로 복사
                    sp500_src = self.data_dir / 'raw' / f"sp500_{target_date_str}.csv"
                    if sp500_src.exists():
                        sp500_dst = self.data_dir / 'processed' / f"sp500_{target_date_str}.csv"
                        shutil.copy(str(sp500_src), str(sp500_dst))
                        self.logger.info(f"Copied S&P 500 file to processed directory")
                except Exception as e:
                    self.logger.warning(f"S&P 500 check failed: {str(e)}")
            else:
                self.logger.info("Step 2: S&P 500 check not needed, skipping...")
            
            # 3. 데이터 처리
            self.logger.info("Step 3: Processing final dataset...")
            self._process_data(target_date, apply_tstat_filter, confidence_level, min_trading_days)
            
            self.logger.info("Pipeline completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            # 파이프라인 예외 추적 정보 출력
            import traceback
            self.logger.error(traceback.format_exc())
            raise
            
    def _process_data(
        self,
        target_date: str,
        apply_tstat_filter: bool = True,
        confidence_level: float = 0.95,
        min_trading_days: int = 1000
    ) -> None:
        """
        데이터 처리 및 필터링을 수행합니다.

        Args:
            target_date: 기준 날짜
            apply_tstat_filter: t-통계량 필터링 적용 여부
            confidence_level: 신뢰 수준
            min_trading_days: 최소 거래일 수
        """
        try:
            # S&P 500 인덱스 데이터 다운로드 및 저장
            self._download_snp500_index()
            
            # 메타데이터 파일 목록 (처리에서 제외할 파일들)
            metadata_files = ['symbol_permno.csv']
            
            # 처리할 파일 목록 가져오기
            raw_files = [f for f in os.listdir(self.data_dir / 'raw') if f.endswith('.csv') and f not in metadata_files]
            
            # raw 폴더의 개별 파일들을 읽어 Data.csv 파일 생성
            raw_dir = self.data_dir / 'raw'
            processed_dir = self.data_dir / 'processed'
            data_path = processed_dir / 'Data.csv'
            filtered_data_path = processed_dir / 'Data_filtered.csv'
            
            # symbol_permno.csv 파일 확인
            symbol_permno_path = processed_dir / 'symbol_permno.csv'
            if not symbol_permno_path.exists():
                self.logger.warning("symbol_permno.csv not found, creating empty mapping")
                symbol_permno = pd.DataFrame(columns=['Symbol', 'PERMNO'])
                
                # 티커 파일들을 읽어서 PERMNO 할당
                ticker_files = list(raw_dir.glob('*.csv'))
                permno_counter = 1
                
                for ticker_file in ticker_files:
                    symbol = ticker_file.stem
                    new_row = pd.DataFrame({'Symbol': [symbol], 'PERMNO': [permno_counter]})
                    symbol_permno = pd.concat([symbol_permno, new_row], ignore_index=True)
                    permno_counter += 1
                    
                symbol_permno.to_csv(symbol_permno_path, index=False)
                self.logger.info(f"Created symbol_permno.csv with {len(symbol_permno)} symbols")
            else:
                symbol_permno = pd.read_csv(symbol_permno_path)
            
            # 개별 파일들을 읽어서 Data.csv 파일 생성
            try:
                self.logger.info("Combining individual stock files into Data.csv...")
                
                # 결과 데이터프레임 초기화
                all_data = []
                
                # 각 티커 파일 처리
                # 메타데이터 파일 제외
                excluded_files = ['symbol_permno.csv', f'sp500_{target_date.replace("-", "")}.csv', 'sp500_ticker.csv']
                ticker_files = [f for f in raw_dir.glob('*.csv') if f.name not in excluded_files]
                self.logger.info(f"Processing {len(ticker_files)} ticker files")
                
                for ticker_file in ticker_files:
                    symbol = ticker_file.stem
                    
                    try:
                        # 티커 파일 읽기
                        ticker_data = pd.read_csv(ticker_file)
                        
                        # 소수점 3째 자리에서 반올림
                        numeric_cols = ticker_data.select_dtypes(include=['float64']).columns
                        for col in numeric_cols:
                            ticker_data[col] = ticker_data[col].round(2)
                        
                        # PERMNO 매핑
                        permno = None
                        symbol_row = symbol_permno[symbol_permno['Symbol'] == symbol]
                        if not symbol_row.empty:
                            permno = symbol_row['PERMNO'].values[0]
                        else:
                            # 새로운 PERMNO 할당
                            permno = symbol_permno['PERMNO'].max() + 1 if not symbol_permno.empty else 1
                            new_row = pd.DataFrame({'Symbol': [symbol], 'PERMNO': [permno]})
                            symbol_permno = pd.concat([symbol_permno, new_row], ignore_index=True)
                        
                        # 필요한 열 추가
                        if 'date' not in ticker_data.columns and 'Date' in ticker_data.columns:
                            ticker_data['date'] = ticker_data['Date']
                        
                        # 필수 열 확인
                        required_cols = ['date', 'Close']
                        if not all(col in ticker_data.columns for col in required_cols):
                            self.logger.warning(f"Skipping {symbol}: missing required columns")
                            continue
                        
                        # 날짜 형식 표준화 (ISO 형식으로 변환)
                        try:
                            ticker_data['date'] = pd.to_datetime(ticker_data['date']).dt.strftime('%Y-%m-%d')
                        except Exception as e:
                            self.logger.warning(f"Error converting date for {symbol}: {str(e)}")
                        
                        # 데이터 변환
                        ticker_processed = pd.DataFrame({
                            'date': ticker_data['date'],
                            'PERMNO': permno,
                            'PRC': ticker_data['Close'].round(2),
                            'VOL': ticker_data.get('Volume', np.nan),
                            'OPENPRC': ticker_data.get('Open', np.nan).round(2) if 'Open' in ticker_data else np.nan,
                            'BIDLO': ticker_data.get('Low', np.nan).round(2) if 'Low' in ticker_data else np.nan,
                            'ASKHI': ticker_data.get('High', np.nan).round(2) if 'High' in ticker_data else np.nan,
                            'RET': ticker_data.get('Return', np.nan).round(4) if 'Return' in ticker_data else np.nan
                        })
                        
                        all_data.append(ticker_processed)
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing {symbol}: {str(e)}")
                
                # 모든 데이터 결합
                if all_data:
                    combined_data = pd.concat(all_data, ignore_index=True)
                    
                    # 날짜 형식 변환 (ISO 형식으로 저장)
                    combined_data['date'] = pd.to_datetime(combined_data['date'], errors='coerce').dt.strftime('%Y-%m-%d')
                    combined_data = combined_data.dropna(subset=['date'])  # 날짜 변환 실패한 행 제거
                    
                    # 소수점 반올림 적용
                    numeric_cols = combined_data.select_dtypes(include=['float64']).columns
                    for col in numeric_cols:
                        if col == 'RET':  # 수익률은 4자리까지 유지
                            combined_data[col] = combined_data[col].round(4)
                        else:  # 가격 데이터는 2자리까지 유지
                            combined_data[col] = combined_data[col].round(2)
                    
                    # 저장
                    combined_data.to_csv(data_path, index=False)
                    self.logger.info(f"Combined data saved to {data_path}")
                    self.logger.info(f"Applied rounding to reduce file size")
                    
                    # symbol_permno.csv 업데이트
                    symbol_permno.to_csv(symbol_permno_path, index=False)
                    self.logger.info(f"Updated symbol_permno.csv with {len(symbol_permno)} symbols")
                    
                    # t-통계량 필터링 적용
                    if apply_tstat_filter:
                        self.logger.info("Applying t-statistic filtering...")
                        
                        # 수익률 계산이 필요한 경우
                        if 'RET' not in combined_data.columns or combined_data['RET'].isna().all():
                            self.logger.info("Calculating returns from price data...")
                            
                            # 날짜를 datetime으로 변환하여 정렬
                            temp_df = combined_data.copy()
                            temp_df['date'] = pd.to_datetime(temp_df['date'])
                            temp_df = temp_df.sort_values(['PERMNO', 'date'])
                            
                            # 종목별로 수익률 계산
                            temp_df['RET'] = temp_df.groupby('PERMNO')['PRC'].pct_change().round(4)
                            
                            # 날짜를 다시 문자열로 변환
                            temp_df['date'] = temp_df['date'].dt.strftime('%Y-%m-%d')
                            
                            # 원본 데이터프레임에 수익률 추가
                            combined_data = temp_df
                            
                            # 업데이트된 데이터 저장
                            combined_data.to_csv(data_path, index=False)
                            self.logger.info(f"Updated Data.csv with calculated returns")
                        
                        try:
                            # t-통계량 필터링 적용
                            filtered_data = self.processor.filter_stocks(
                                input_file=data_path,
                                output_file=filtered_data_path,
                                min_trading_days=min_trading_days,
                                confidence_level=confidence_level
                            )
                            
                            # 필터링된 데이터에도 소수점 반올림 적용
                            if filtered_data is not None and not filtered_data.empty:
                                numeric_cols = filtered_data.select_dtypes(include=['float64']).columns
                                for col in numeric_cols:
                                    if col == 'RET':  # 수익률은 4자리까지 유지
                                        filtered_data[col] = filtered_data[col].round(4)
                                    else:  # 가격 데이터는 2자리까지 유지
                                        filtered_data[col] = filtered_data[col].round(2)
                                
                                # 업데이트된 필터링 데이터 저장
                                filtered_data.to_csv(filtered_data_path, index=False)
                            
                            self.logger.info(f"T-statistic filtered data saved to {filtered_data_path}")
                            self.logger.info(f"Original stocks: {len(combined_data['PERMNO'].unique())}, "
                                            f"Filtered stocks: {len(filtered_data['PERMNO'].unique())}")
                        except Exception as e:
                            self.logger.error(f"Error in t-statistic filtering: {str(e)}")
                            self.logger.info("Continuing without t-statistic filtering...")
                else:
                    self.logger.error("No valid data found in ticker files")
                    raise ValueError("No valid data found in ticker files")
                
            except Exception as e:
                self.logger.error(f"Error processing data: {str(e)}")
                raise
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            raise

    def _download_snp500_index(self) -> None:
        """S&P 500 인덱스 데이터를 다운로드하고 저장합니다."""
        try:
            # 파일 경로 설정
            snp500_path = os.path.join(self.data_dir / 'processed', "snp500_index.csv")
            
            # 이미 파일이 존재하는지 확인
            if os.path.exists(snp500_path):
                self.logger.info(f"S&P 500 인덱스 데이터가 이미 존재합니다: {snp500_path}")
                return
                
            # SPY ETF 데이터 다운로드 (전체 기간)
            self.logger.info("S&P 500 인덱스 데이터 다운로드 중...")
            spy = yf.Ticker("SPY")
            spy_data = spy.history(period="max")
            
            # 컬럼명 변경 (필요한 경우)
            if 'Close' not in spy_data.columns and 'close' in spy_data.columns:
                spy_data.rename(columns={'close': 'Close'}, inplace=True)
            if 'Adj Close' not in spy_data.columns and 'adjclose' in spy_data.columns:
                spy_data.rename(columns={'adjclose': 'Adj Close'}, inplace=True)
            
            # 데이터 저장
            spy_data.reset_index().to_csv(snp500_path, index=False)
            self.logger.info(f"S&P 500 인덱스 데이터가 저장되었습니다: {snp500_path}")
            
        except Exception as e:
            self.logger.error(f"S&P 500 인덱스 데이터 다운로드 중 오류 발생: {str(e)}")
            raise

def main():
    """파이프라인 실행을 위한 메인 함수"""
    pipeline = DataPipeline()
    pipeline.run_pipeline()

if __name__ == "__main__":
    main() 