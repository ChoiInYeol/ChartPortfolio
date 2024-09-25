import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. 데이터 파일 불러오기
def load_sp500_data(file_path):
    """S&P 500 구성 종목 데이터를 불러옵니다."""
    df = pd.read_csv(file_path, parse_dates=['date'])
    df['tickers'] = df['tickers'].apply(lambda x: x.split(','))
    return df

# 2. 데이터 전처리
def preprocess_data(df):
    """데이터를 전처리합니다."""
    all_tickers = set()
    for tickers in df['tickers']:
        all_tickers.update(tickers)
    return list(all_tickers), df

# 3. 각 종목별 SPY 지수 포함 여부 및 상태 추적
def track_ticker_status(df, all_tickers):
    """각 종목의 SPY 지수 포함 여부와 상태를 추적합니다."""
    ticker_status = {ticker: [] for ticker in all_tickers}
    
    for _, row in df.iterrows():
        date = row['date']
        current_tickers = set(row['tickers'])
        
        for ticker in all_tickers:
            if ticker in current_tickers:
                status = 'Included'
            elif not ticker_status[ticker] or ticker_status[ticker][-1][1] != 'Excluded':
                status = 'Excluded'
            else:
                status = ticker_status[ticker][-1][1]
            
            ticker_status[ticker].append((date, status))
    
    return ticker_status

# 4. 최종 데이터셋 생성
def create_final_dataset(ticker_status):
    """최종 데이터셋을 생성합니다."""
    data = []
    for ticker, status_list in ticker_status.items():
        for i, (date, status) in enumerate(status_list):
            if i == 0 or status != status_list[i-1][1]:
                data.append({
                    'ticker': ticker,
                    'date': date,
                    'status': status
                })
    
    return pd.DataFrame(data)

# 메인 실행 코드
if __name__ == "__main__":
    file_path = "S&P 500 Historical Components & Changes(08-17-2024).csv"
    df = load_sp500_data(file_path)
    all_tickers, df = preprocess_data(df)
    ticker_status = track_ticker_status(df, all_tickers)
    final_dataset = create_final_dataset(ticker_status)
    
    print(final_dataset.head(10))
    final_dataset.to_csv("sp500_ticker_status_1996_2024.csv", index=False)