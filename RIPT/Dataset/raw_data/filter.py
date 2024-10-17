import pandas as pd
import numpy as np
import os

def handle_outliers(df, lookback=5, tolerance=0.05, volatility_threshold=2.0):
    """
    주가와 수익률 데이터를 사용해 이상치를 처리하고, 변동성이 극심한 주식은 제거합니다.
    """
    high_threshold = 0.75
    low_threshold = -0.75

    # 수익률 계산
    df['return'] = df.groupby('PERMNO')['PRC'].pct_change()

    # 극단적인 수익률을 가진 주식 탐지
    extreme_returns = df[(df['return'] > high_threshold) | (df['return'] < low_threshold)]

    # 변동성이 높은 주식 식별
    max_daily_volatility = df.groupby('PERMNO')['return'].apply(lambda x: x.abs().max())
    stocks_to_remove = max_daily_volatility[max_daily_volatility > volatility_threshold].index

    print(f"총 {len(stocks_to_remove)}개의 주식이 변동성 기준을 초과하여 제거됩니다.")
    print(f"제거된 주식 목록: {list(stocks_to_remove)}")

    # 변동성이 높은 주식 제거
    df = df[~df['PERMNO'].isin(stocks_to_remove)]

    # 극단치 처리
    to_adjust = []
    for _, group in df.groupby('PERMNO'):
        extreme_group = group[group['return'].abs() > high_threshold]
        for idx in extreme_group.index:
            start_idx = max(0, group.index.get_loc(idx) - lookback)
            end_idx = min(len(group) - 1, group.index.get_loc(idx) + lookback)
            price_window = group.iloc[start_idx:end_idx + 1]['PRC']
            if len(price_window) < 2:
                continue
            first_price = price_window.iloc[0]
            last_price = price_window.iloc[-1]
            price_change = abs(last_price - first_price) / first_price
            if price_change <= tolerance:
                to_adjust.append(idx)

    # 수익률 조정
    df.loc[to_adjust, 'return'] = 0

    # 기타 이상치에 대한 보간 처리
    def clean_returns(x):
        return x.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    df['return'] = df.groupby('PERMNO')['return'].transform(clean_returns)

    print(f"총 {len(to_adjust)}개의 이상치 수익률을 조정했습니다.")
    print(f"총 {df['PERMNO'].nunique()}개의 주식에 대해 수익률 조정을 수행했습니다.")

    return df

def filter_stocks(
    input_file='RIPT/Dataset/raw_data/all_data.csv', 
    output_file='filtered_stock.csv', 
    min_trading_days=1000,
    start_date='2001-01-01', 
    end_date='2024-09-01'
):
    """
    Trims stock data to the specified date range, filters stocks with at least a given number of trading days,
    and handles outliers.
    """
    # Read the input CSV file and parse dates
    df = pd.read_csv(input_file, parse_dates=['date'])

    # Trim data to the specified date range
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    # Calculate the number of trading days per stock in the trimmed data
    trading_days = df.groupby('PERMNO')['date'].count()

    # Filter stocks with at least the minimum trading days
    valid_permnos = trading_days[trading_days >= min_trading_days].index
    df = df[df['PERMNO'].isin(valid_permnos)]

    print(f"Original number of stocks: {df['PERMNO'].nunique()}")

    # Handle outliers
    df = handle_outliers(df)

    # Save the filtered and cleaned data to a new CSV file
    df.to_csv(output_file, index=False)

    print(f"Final number of stocks: {df['PERMNO'].nunique()}")
    print(f"Filtered and cleaned data saved to {output_file}.")

if __name__ == "__main__":
    filter_stocks()
