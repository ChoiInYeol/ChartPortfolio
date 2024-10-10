"""
주식 데이터 다운로드 및 전처리 모듈

이 모듈은 Yahoo Finance API를 사용하여 주식 데이터를 다운로드하고 전처리합니다.
주요 기능으로는 개별 주식 데이터 다운로드, 데이터 전처리, 그리고 전체 주식 목록에 대한 데이터 다운로드가 있습니다.
"""

import os
import pandas as pd
import yfinance as yf
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import warnings

# 경고 무시 설정
warnings.simplefilter(action="ignore", category=FutureWarning)

# 로깅 설정
logging.basicConfig(
    filename='stock_download.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_stock_data(ticker, start_date, end_date):
    """
    지정된 주식 티커에 대한 데이터를 다운로드합니다.

    Args:
        ticker (str): 주식 티커 심볼
        start_date (str): 데이터 시작 날짜 (YYYY-MM-DD 형식)
        end_date (str): 데이터 종료 날짜 (YYYY-MM-DD 형식)

    Returns:
        pd.DataFrame or None: 다운로드된 주식 데이터 또는 오류 발생 시 None
    """
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            prepost=False,
            threads=True,
            proxy=None
        )
        if data.empty:
            logging.warning(f"{ticker}에 대한 데이터를 찾을 수 없습니다. 건너뜁니다.")
            return None
        return data
    except Exception as e:
        logging.error(f"{ticker} 다운로드 중 오류 발생: {e}")
        return None

def preprocess_stock_data(data, ticker):
    """
    다운로드된 주식 데이터를 전처리합니다.

    Args:
        data (pd.DataFrame): 원본 주식 데이터
        ticker (str): 주식 티커 심볼

    Returns:
        pd.DataFrame: 전처리된 주식 데이터
    """
    data['TICKER'] = ticker
    data['date'] = data.index.strftime('%Y%m%d')
    data = data.reset_index()
    data = data.rename(columns={
        'Date': 'date',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Adj Close': 'Adj Close',
        'Volume': 'Vol'
    })
    return data

def download_and_process_stock(ticker, start_date, end_date, output_dir):
    """
    주식 데이터를 다운로드하고 전처리한 후 저장합니다.

    Args:
        ticker (str): 주식 티커 심볼
        start_date (str): 데이터 시작 날짜
        end_date (str): 데이터 종료 날짜
        output_dir (str): 출력 디렉토리 경로

    Returns:
        bool: 성공 시 True, 실패 시 False
    """
    output_file = os.path.join(output_dir, f"{ticker}.csv")
    if os.path.exists(output_file):
        logging.info(f"{ticker} 파일이 이미 존재합니다. 다운로드를 건너뜁니다.")
        return True

    data = download_stock_data(ticker, start_date, end_date)
    if data is None or data.empty:
        return False

    processed_data = preprocess_stock_data(data, ticker)
    processed_data.to_csv(output_file, index=False)
    logging.info(f"{ticker} 데이터가 성공적으로 다운로드 및 저장되었습니다.")
    return True

def download_all_stocks(tickers, start_date, end_date, output_dir):
    """
    주어진 모든 주식 티커에 대해 데이터를 다운로드하고 처리합니다.

    Args:
        tickers (list): 주식 티커 심볼 리스트
        start_date (str): 데이터 시작 날짜
        end_date (str): 데이터 종료 날짜
        output_dir (str): 출력 디렉토리 경로

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(
                download_and_process_stock, ticker, start_date, end_date, output_dir
            ): ticker for ticker in tickers
        }
        for future in tqdm(as_completed(futures), total=len(tickers), desc="주식 다운로드 중"):
            ticker = futures[future]
            try:
                success = future.result()
                if not success:
                    logging.warning(f"{ticker}에 대한 충분한 데이터를 다운로드하지 못했습니다.")
            except Exception as e:
                logging.error(f"{ticker} 처리 중 예외 발생: {e}")

def main():
    """
    메인 함수: 주식 데이터 다운로드 프로세스를 실행합니다.
    """
    # 티커 파일에서 주식 목록 읽기
    ticker_df = pd.read_csv('ticker.csv')
    tickers = ticker_df['Symbol'].unique().tolist()

    start_date = "2000-01-01"
    end_date = "2023-12-31"
    output_dir = "./data/stocks/"

    download_all_stocks(tickers, start_date, end_date, output_dir)

if __name__ == "__main__":
    main()