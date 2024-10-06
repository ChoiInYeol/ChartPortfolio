"""
주식 데이터 전처리 스크립트

이 스크립트는 data/stocks 디렉토리에 있는 모든 CSV 파일을 처리합니다.
모든 주식 데이터의 날짜 범위를 통일시키고, 누락된 데이터를 -1로 채웁니다.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

# 로깅 설정
logging.basicConfig(filename='preprocessing.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_stock_data(stock_dir: str = "data/stocks/"):
    """
    주식 데이터를 전처리합니다.

    Args:
        stock_dir (str): 주식 데이터가 저장된 디렉토리 경로

    Returns:
        None
    """
    # 모든 CSV 파일 목록 가져오기
    csv_files = [f for f in os.listdir(stock_dir) if f.endswith('.csv')]

    # 모든 파일의 시작 날짜와 종료 날짜 찾기
    all_dates = pd.DatetimeIndex([])
    for file in csv_files:
        df = pd.read_csv(os.path.join(stock_dir, file), parse_dates=['Date'])
        all_dates = all_dates.union(df['Date'])

    start_date = all_dates.min()
    end_date = all_dates.max()
    date_range = pd.date_range(start=start_date, end=end_date)

    logging.info(f"전체 날짜 범위: {start_date} ~ {end_date}")

    # 각 CSV 파일 처리
    for file in tqdm(csv_files, desc="파일 처리 중"):
        file_path = os.path.join(stock_dir, file)
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df.set_index('Date', inplace=True)

        # 새로운 날짜 범위로 데이터프레임 재인덱싱
        df = df.reindex(date_range)

        # 누락된 데이터를 -1로 채우기
        df.fillna(-1, inplace=True)

        # 처리된 데이터를 원래 파일에 덮어쓰기
        df.to_csv(file_path)

    logging.info("모든 주식 데이터 전처리 완료")

if __name__ == "__main__":
    preprocess_stock_data()