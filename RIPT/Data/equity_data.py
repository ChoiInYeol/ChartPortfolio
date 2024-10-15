from __future__ import print_function, division
import os
import os.path as op
import numpy as np
import pandas as pd
import time
from Data import dgp_config as dcf

def safe_exp(x):
    return np.exp(np.clip(x, -10, 10))

def create_spy_returns():
    spy = pd.read_csv('data/spy.csv', parse_dates=['Date'], index_col='Date')
    spy['Return'] = spy['Adj Close'].pct_change()

    for freq in ['week', 'month', 'quarter', 'year']:
        if freq == 'week':
            returns = spy['Return'].resample('W').apply(lambda x: (1 + x).prod() - 1)
        elif freq == 'month':
            returns = spy['Return'].resample('ME').apply(lambda x: (1 + x).prod() - 1)
        elif freq == 'quarter':
            returns = spy['Return'].resample('QE').apply(lambda x: (1 + x).prod() - 1)
        else:  # year
            returns = spy['Return'].resample('YE').apply(lambda x: (1 + x).prod() - 1)
        
        returns = pd.DataFrame(returns, columns=['Return'])
        returns['nxt_freq_ewret'] = returns['Return'].shift(-1)  # 다음 기간의 수익률 추가
        
        output_path = os.path.join(dcf.CACHE_DIR, f"spy_{freq}_ret.csv")
        returns.to_csv(output_path, index=True)
        print(f"Saved spy_{freq}_ret.csv")

def create_benchmark_returns():
    """
    벤치마크 수익률을 계산하고 저장합니다.
    
    0인 데이터를 제외하고 유효한 값만을 사용하여 1/N 방식으로 수익률을 계산합니다.
    주간, 월간, 분기별, 연간 수익률을 계산하여 각각 CSV 파일로 저장합니다.
    """
    path = op.join(dcf.RAW_DATA_DIR, "filtered_stock_1985.csv")
    df = pd.read_csv(path, parse_dates=['date'])
    df.set_index('date', inplace=True)
    
    # date 를 Date로 변경
    df.index.name = 'Date'
    
    df['Return'] = df.groupby('PERMNO')['PRC'].pct_change()
    
    for freq in ['week', 'month', 'quarter', 'year']:
        if freq == 'week':
            returns = df.groupby([pd.Grouper(freq='W'), 'PERMNO'])['Return'].apply(lambda x: (1 + x).prod() - 1)
        elif freq == 'month':
            returns = df.groupby([pd.Grouper(freq='ME'), 'PERMNO'])['Return'].apply(lambda x: (1 + x).prod() - 1)
        elif freq == 'quarter':
            returns = df.groupby([pd.Grouper(freq='QE'), 'PERMNO'])['Return'].apply(lambda x: (1 + x).prod() - 1)
        else:  # year
            returns = df.groupby([pd.Grouper(freq='YE'), 'PERMNO'])['Return'].apply(lambda x: (1 + x).prod() - 1)
        
        # 0이 아닌 유효한 값만 선택하여 1/N 방식으로 평균 계산
        benchmark_returns = returns.groupby(level=0).apply(lambda x: x[x != 0].mean())
        
        benchmark_returns = pd.DataFrame(benchmark_returns, columns=['Return'])
        benchmark_returns['nxt_freq_ewret'] = benchmark_returns['Return'].shift(-1)  # 다음 기간의 수익률 추가
        
        output_path = os.path.join(dcf.CACHE_DIR, f"benchmark_{freq}_ret.csv")
        benchmark_returns.to_csv(output_path, index=True)
        print(f"{freq} 벤치마크 수익률이 {output_path}에 저장되었습니다.")

def get_spy_freq_rets(freq):
    assert freq in ["week", "month", "quarter", 'year'], f"Invalid freq: {freq}"
    file_path = os.path.join(dcf.CACHE_DIR, f"spy_{freq}_ret.csv")
    
    if not os.path.exists(file_path):
        create_return_files()
    
    spy = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return spy

def get_bench_freq_rets(freq):
    assert freq in ["week", "month", "quarter", "year"]
    file_path = os.path.join(dcf.CACHE_DIR, f"benchmark_{freq}_ret.csv")
    
    if not os.path.exists(file_path):
        create_return_files()
    
    bench = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return bench

def create_return_files():
    os.makedirs(dcf.CACHE_DIR, exist_ok=True)
    create_spy_returns()
    create_benchmark_returns()

def get_period_end_dates(period):
    assert period in ["week", "month", "quarter", "year"]
    spy = get_spy_freq_rets(period)
    return spy.index

def processed_US_data():
    processed_us_data_path = op.join(dcf.PROCESSED_DATA_DIR, "us_ret.feather")
    if op.exists(processed_us_data_path):
        print(f"Loading processed data from {processed_us_data_path}")
        since = time.time()
        df = pd.read_feather(processed_us_data_path)
        df.set_index(["Date", "StockID"], inplace=True)
        df.sort_index(inplace=True)
        print(f"Finish loading processed data in {(time.time() - since) / 60:.2f} min")
        return df.copy()

    raw_us_data_path = op.join(dcf.RAW_DATA_DIR, "filtered_stock_1985.csv")
    print("Reading raw data from {}".format(raw_us_data_path))
    since = time.time()
    df = pd.read_csv(
        raw_us_data_path,
        parse_dates=["date"],
        dtype={
            "PERMNO": str,
            "BIDLO": np.float64,
            "ASKHI": np.float64,
            "PRC": np.float64,
            "VOL": np.float64,
            "SHROUT": np.float64,
            "OPENPRC": np.float64,
            "RET": object,
            "EXCHCD": np.float64,
        },
        #compression="gzip",
        header=0,
    )
    print(f"finish reading data in {(time.time() - since) / 60:.2f} s")
    df = process_raw_data_helper(df)

    df.reset_index().to_feather(processed_us_data_path)
    return df.copy()

def process_raw_data_helper(df):
    df = df.rename(
        columns={
            "date": "Date",
            "PERMNO": "StockID",
            "BIDLO": "Low",
            "ASKHI": "High",
            "PRC": "Close",
            "VOL": "Vol",
            "SHROUT": "Shares",
            "OPENPRC": "Open",
            "RET": "Ret",
        }
    )
    
    df.StockID = df.StockID.astype(str)
    df.Ret = df.Ret.astype(str)
    df = df.replace(
        {
            "Close": {0: np.nan},
            "Open": {0: np.nan},
            "High": {0: np.nan},
            "Low": {0: np.nan},
            "Ret": {"C": np.nan, "B": np.nan, "A": np.nan, ".": np.nan},
            "Vol": {0: np.nan, (-99): np.nan},
        }
    )
    if "Shares" not in df.columns:
        df["Shares"] = 0
    df["Ret"] = df.Ret.astype(np.float64)
    df = df.dropna(subset=["Ret"])
    df[["Close", "Open", "High", "Low", "Vol", "Shares"]] = df[
        ["Close", "Open", "High", "Low", "Vol", "Shares"]
    ].abs()
    df["MarketCap"] = np.abs(df["Close"] * df["Shares"])
    df.set_index(["Date", "StockID"], inplace=True)
    df.sort_index(inplace=True)
    
    df['Ret'] = df['Ret'].apply(lambda x: max(x, -0.9999))
    df["log_ret"] = np.log(1 + df.Ret)
    df["cum_log_ret"] = df.groupby("StockID")["log_ret"].cumsum(skipna=True)
    df["EWMA_vol"] = df.groupby("StockID")["Ret"].transform(
        lambda x: (x**2).ewm(alpha=0.05).mean().shift(periods=1)
    )
    
    for freq in ["week", "month", "quarter", "year"]:
        period_end_dates = get_period_end_dates(freq)
        freq_df = df[df.index.get_level_values("Date").isin(period_end_dates)].copy()
        
        freq_df["freq_ret"] = freq_df.groupby("StockID")["cum_log_ret"].transform(
            lambda x: np.exp(x.shift(-1) - x) - 1
        )
        
        print(
            f"Freq: {freq}: {len(freq_df)}/{len(df)} preriod_end_dates from \
                        {period_end_dates[0]}, {period_end_dates[1]},  to {period_end_dates[-1]}"
        )
        
        df[f"Ret_{freq}"] = freq_df["freq_ret"]
        num_nan = np.sum(pd.isna(df[f"Ret_{freq}"]))
        print(f"df Ret_{freq} {len(df) - num_nan}/{len(df)} not nan")
        
    for i in [5, 20, 60, 65, 180, 250, 260]:
        print(f"Calculating {i}d return")
        df[f"Ret_{i}d"] = df.groupby("StockID")["cum_log_ret"].transform(
            lambda x: np.exp(x.shift(-i) - x) - 1
        )
    return df

def get_processed_US_data_by_year(year):
    df = processed_US_data()
    
    if not isinstance(df.index.get_level_values(0)[0], pd.Timestamp):
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index(['Date', 'StockID'], inplace=True)
    
    df = df[
        df.index.get_level_values(0).year.isin([year, year - 1, year - 2])
    ].copy()
    
    return df

def get_period_ret(period, country="USA"):
    assert country == "USA"
    assert period in ["week", "month", "quarter"]
    period_ret_path = op.join(dcf.CACHE_DIR, f"us_{period}_ret.pq")
    period_ret = pd.read_parquet(period_ret_path)
    period_ret.set_index(["Date", "StockID"], inplace=True)
    period_ret.sort_index(inplace=True)
    return period_ret

def calculate_next_ret(df, freq):
    freq_map = {'day': 1, 'week': 5, 'month': 20, 'quarter': 60, 'year': 250}
    if freq not in freq_map:
        raise ValueError('Invalid frequency')
    shift_value = freq_map[freq]
    df[f'next_{freq}_ret'] = df.groupby('StockID')['Ret'].shift(-shift_value)
    return df

def calculate_next_ret_delay(df, freq, d):
    freq_map = {'day': 1, 'week': 5, 'month': 20, 'quarter': 60, 'year': 250}
    if freq not in freq_map:
        raise ValueError('Invalid frequency')
    shift_value = freq_map[freq] + d
    df[f'next_{freq}_ret_{d}delay'] = df.groupby('StockID')['Ret'].shift(-shift_value)
    return df

def calculate_ret(df, start, end):
    df[f'Ret_{start}-{end}d'] = df.groupby('StockID')['Ret'].rolling(end - start + 1).sum().reset_index(0, drop=True)
    return df

def create_period_ret_file(freq):
    stock_path = op.join(dcf.PROCESSED_DATA_DIR, "us_ret.feather")
    if not op.exists(stock_path):
        processed_US_data()  # 필요한 경우 데이터 생성
    
    stock = pd.read_feather(stock_path)
    stock = stock[['Date', 'StockID', 'MarketCap', 'Close', 'Ret']]
    
    us_freq_ret = stock.copy()
    us_freq_ret = calculate_next_ret(us_freq_ret, freq)
    us_freq_ret = calculate_next_ret_delay(us_freq_ret, freq, 0)
    us_freq_ret = calculate_ret(us_freq_ret, 6, 20)
    us_freq_ret = calculate_ret(us_freq_ret, 6, 60)
    
    period_ret_path = op.join(dcf.CACHE_DIR, f"us_{freq}_ret.pq")
    us_freq_ret.to_parquet(period_ret_path)
    print(f"Created {period_ret_path}")

def get_period_ret(period, country="USA"):
    assert country == "USA"
    assert period in ["week", "month", "quarter"]
    period_ret_path = op.join(dcf.CACHE_DIR, f"us_{period}_ret.pq")
    
    if not op.exists(period_ret_path):
        print(f"{period_ret_path} not found. Creating it now.")
        create_period_ret_file(period)
    
    period_ret = pd.read_parquet(period_ret_path)
    period_ret.set_index(["Date", "StockID"], inplace=True)
    period_ret.sort_index(inplace=True)
    return period_ret

if __name__ == "__main__":
    create_return_files()
    for freq in ["week", "month", "quarter"]:
        create_period_ret_file(freq)