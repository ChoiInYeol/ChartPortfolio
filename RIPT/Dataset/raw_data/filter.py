import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime

def setup_logging(log_dir='logs'):
    """
    로깅 설정을 초기화합니다.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'stock_filter_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.info("로깅 시작")
    return log_file

def detect_abnormal_stocks(returns, prices, price_jumps_threshold=0.5, recovery_window=5, price_gap_threshold=10):
    """
    주가 데이터의 이상치를 탐지하는 향상된 함수입니다.
    
    Args:
        returns (pd.DataFrame): 일별 수익률 데이터
        prices (pd.DataFrame): 일별 주가 데이터
        price_jumps_threshold (float): 가격 점프 임계값
        recovery_window (int): 가격 회복을 확인할 기간 (일)
        price_gap_threshold (float): 연속된 거래일 사이의 최대 허용 가격 변동 비율
        
    Returns:
        List: 제거할 종목의 목록
    """
    stocks_to_remove = set()
    
    # 1. 급격한 가격 하락 후 빠른 회복 패턴 탐지
    for stock in prices.columns:
        price_series = prices[stock].dropna()
        if len(price_series) < recovery_window:
            continue
            
        for i in range(len(price_series) - recovery_window):
            window = price_series.iloc[i:i+recovery_window]
            initial_price = window.iloc[0]
            min_price = window.min()
            final_price = window.iloc[-1]
            
            # 급격한 하락 후 빠른 회복 패턴 확인
            if (min_price < initial_price * 0.1 and  # 90% 이상 하락
                final_price > initial_price * 0.8):  # 초기가의 80% 이상으로 회복
                stocks_to_remove.add(stock)
                break
    
    # 2. 연속된 거래일 간의 비정상적인 가격 변동 탐지
    for stock in prices.columns:
        price_series = prices[stock].dropna()
        daily_changes = price_series.pct_change().abs()
        
        # 일일 변동폭이 임계값을 초과하는 경우 확인
        if (daily_changes > price_gap_threshold).any():
            stocks_to_remove.add(stock)
    
    # 3. 이동평균을 이용한 이상치 탐지
    window_size = 20
    for stock in prices.columns:
        price_series = prices[stock].dropna()
        if len(price_series) < window_size:
            continue
            
        rolling_mean = price_series.rolling(window=window_size).mean()
        rolling_std = price_series.rolling(window=window_size).std()
        
        # 가격이 평균에서 표준편차의 5배 이상 벗어나는 경우
        z_scores = (price_series - rolling_mean) / rolling_std
        if (abs(z_scores) > 5).any():
            stocks_to_remove.add(stock)
    
    logging.info(f"이상치 탐지 결과:")
    logging.info(f"- 총 제거 대상 종목 수: {len(stocks_to_remove)}개")
    
    return list(stocks_to_remove)

def filter_stocks(
    input_file='RIPT/Dataset/raw_data/all_data.csv', 
    output_file='filtered_stock.csv', 
    min_trading_days=1000,
    start_date='2001-01-01', 
    end_date='2024-09-01',
    price_jumps_threshold=0.5,
    recovery_window=5,
    price_gap_threshold=10
):
    """
    이상치 데이터를 가진 종목을 제거하여 주식 데이터를 필터링합니다.
    
    Args:
        input_file (str): 입력 파일 경로
        output_file (str): 출력 파일 경로
        min_trading_days (int): 최소 거래일수
        start_date (str): 시작일
        end_date (str): 종료일
        price_jumps_threshold (float): 가격 점프 임계값
        recovery_window (int): 가격 회복 확인 기간
        price_gap_threshold (float): 최대 허용 가격 변동 비율
    """
    try:
        logging.info(f"데이터 필터링 시작: {input_file}")
        logging.info(f"파라미터 - 최소거래일: {min_trading_days}, 시작일: {start_date}, 종료일: {end_date}")
    
        # 데이터 로드 및 날짜 범위 필터링
        df = pd.read_csv(input_file, parse_dates=['date'])
        initial_count = len(df)
        logging.info(f"초기 데이터 수: {initial_count:,}행")
    
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        date_filtered_count = len(df)
        logging.info(f"날짜 필터링 후 데이터 수: {date_filtered_count:,}행 (제거됨: {initial_count - date_filtered_count:,}행)")
    
        # 최소 거래일수 필터링
        trading_days = df.groupby('PERMNO')['date'].count()
        valid_permnos = trading_days[trading_days >= min_trading_days].index
        df = df[df['PERMNO'].isin(valid_permnos)]
        
        logging.info(f"거래일수 기준 필터링 후 주식 수: {len(valid_permnos):,}개")
    
        # 가격과 수익률 데이터 피벗
        prices_df = df.pivot(index='date', columns='PERMNO', values='PRC').abs()
        returns_df = df.pivot(index='date', columns='PERMNO', values='RET')
        
        logging.info(f"피벗 데이터 형태 - 가격: {prices_df.shape}, 수익률: {returns_df.shape}")
    
        # 이상치 종목 탐지 및 제거
        stocks_to_remove = detect_abnormal_stocks(
            returns=returns_df,
            prices=prices_df,
            price_jumps_threshold=price_jumps_threshold,
            recovery_window=recovery_window,
            price_gap_threshold=price_gap_threshold
        )
        
        # 이상치 종목 제거
        df_cleaned = df[~df['PERMNO'].isin(stocks_to_remove)]
        final_count = len(df_cleaned)
        logging.info(f"이상치 종목 제거 후 데이터 수: {final_count:,}행 (제거됨: {initial_count - final_count:,}행)")
    
        # 결과 저장
        df_cleaned.to_csv(output_file, index=False)
        logging.info(f"최종 데이터 저장 완료: {output_file}")
    
        return df_cleaned
    
    except Exception as e:
        logging.error(f"에러 발생: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    log_file = setup_logging()
    logging.info("프로그램 시작")
    
    try:
        df_cleaned = filter_stocks(
            input_file='RIPT/Dataset/raw_data/all_data.csv', 
            output_file='filtered_stock.csv', 
            min_trading_days=1000,
            start_date='2001-01-01',
            end_date='2024-09-01',
            price_jumps_threshold=0.75,
            recovery_window=5,
            price_gap_threshold=10
        )
        logging.info("프로그램 정상 종료")
    except Exception as e:
        logging.error("프로그램 비정상 종료", exc_info=True)
    finally:
        logging.info(f"로그 파일 위치: {log_file}")
