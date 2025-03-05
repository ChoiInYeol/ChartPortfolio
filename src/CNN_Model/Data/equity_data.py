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
    """S&P 500 수익률을 계산하고 parquet 형식으로 저장합니다."""
    # 이미 처리된 파일이 있는지 확인
    all_files_exist = True
    for freq in ['week', 'month', 'quarter', 'year']:
        parquet_path = os.path.join(dcf.CACHE_DIR, f"spy_{freq}_ret.parquet")
        if not os.path.exists(parquet_path):
            all_files_exist = False
            break
    
    # 모든 파일이 이미 존재하면 처리 건너뛰기
    if all_files_exist:
        print("모든 S&P 500 수익률 파일이 이미 존재합니다.")
        return
    
    # CSV 파일 한 번만 읽기
    spy = pd.read_csv(op.join(dcf.FILTERED_DATA_DIR, "snp500_index.csv"), parse_dates=['Date'], index_col='Date')
    
    # 'Adj Close' 컬럼이 없는 경우 처리
    if 'Adj Close' not in spy.columns:
        print("경고: 'Adj Close' 컬럼이 없습니다. 'Close' 컬럼을 사용합니다.")
        if 'adjclose' in spy.columns:
            spy.rename(columns={'adjclose': 'Adj Close'}, inplace=True)
        elif 'Close' in spy.columns:
            spy['Adj Close'] = spy['Close']
        else:
            raise ValueError("S&P 500 데이터에 'Adj Close' 또는 'Close' 컬럼이 없습니다.")
    
    spy['Return'] = spy['Adj Close'].pct_change()
    
    for freq in ['week', 'month', 'quarter', 'year']:
        parquet_path = os.path.join(dcf.CACHE_DIR, f"spy_{freq}_ret.parquet")
        
        # 이미 처리된 파일이 있으면 건너뛰기
        if os.path.exists(parquet_path):
            print(f"{parquet_path} 파일이 이미 존재합니다.")
            continue
        
        if freq == 'week':
            # 주별 마지막 거래일 찾기
            week_ends = spy.groupby(pd.Grouper(freq='W')).apply(lambda x: x.index[-1] if not x.empty else None)
            returns = pd.Series(index=week_ends, dtype=float)
            
            for week_end in week_ends:
                if week_end is not None:
                    week_start = week_end - pd.Timedelta(days=7)
                    week_data = spy.loc[week_start:week_end]
                    returns[week_end] = (1 + week_data['Return']).prod() - 1
                    
        else:
            # 월/분기/연도의 첫날과 마지막날 찾기
            freq_map = {'month': 'ME', 'quarter': 'QE', 'year': 'YE'}
            period_groups = spy.groupby(pd.Grouper(freq=freq_map[freq]))
            
            dates = []
            rets = []
            
            for period, group in period_groups:
                if not group.empty:
                    last_day = group.index.max()
                    dates.append(last_day)
                    rets.append((1 + group['Return']).prod() - 1)
            
            returns = pd.Series(rets, index=pd.DatetimeIndex(dates))
        
        # 결과를 데이터프레임으로 변환
        returns = pd.DataFrame(returns, columns=['Return'])
        returns.index.name = 'Date'
        
        # 다음 기간의 수익률 추가
        returns['nxt_freq_ewret'] = returns['Return'].shift(-1)
        
        # parquet 파일로 저장
        returns.reset_index().to_parquet(parquet_path, index=False)
        print(f"S&P 500 {freq} 수익률이 {parquet_path}에 저장되었습니다.")

def create_benchmark_returns():
    """벤치마크 수익률을 계산하고 parquet 형식으로 저장합니다."""
    # 이미 처리된 파일이 있는지 확인
    all_files_exist = True
    for freq in ['week', 'month', 'quarter', 'year']:
        parquet_path = os.path.join(dcf.CACHE_DIR, f"benchmark_{freq}_ret.parquet")
        if not os.path.exists(parquet_path):
            all_files_exist = False
            break
    
    # 모든 파일이 이미 존재하면 처리 건너뛰기
    if all_files_exist:
        print("모든 벤치마크 수익률 파일이 이미 존재합니다.")
        return
    
    # 벤치마크 데이터 한 번만 로드
    path = op.join(dcf.FILTERED_DATA_DIR, "Data_filtered.csv")
    print(f"벤치마크 데이터 로드 중: {path}")
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.sort_values('date')  # 날짜순으로 정렬
    df['Return'] = df.groupby('PERMNO')['PRC'].pct_change()
    
    # 고유한 거래일 목록 생성
    trading_days = pd.Series(df['date'].unique()).sort_values()
    
    # S&P 500 데이터 로드
    spy_path = op.join(dcf.FILTERED_DATA_DIR, "snp500_index.csv")
    spy = pd.read_csv(spy_path, parse_dates=['Date'])
    spy = spy[spy['Date'] >= trading_days.min()]
    
    for freq in ['week', 'month', 'quarter', 'year']:
        parquet_path = os.path.join(dcf.CACHE_DIR, f"benchmark_{freq}_ret.parquet")
        
        # 이미 처리된 파일이 있으면 건너뛰기
        if os.path.exists(parquet_path):
            print(f"{parquet_path} 파일이 이미 존재합니다.")
            continue
        
        # S&P 500의 기간별 마지막 거래일 가져오기
        spy_returns_path = os.path.join(dcf.CACHE_DIR, f"spy_{freq}_ret.parquet")
        
        # S&P 500 수익률 파일이 없으면 생성
        if not os.path.exists(spy_returns_path):
            create_spy_returns()
        
        # S&P 500 수익률 파일 로드
        spy_returns = pd.read_parquet(spy_returns_path)
        spy_returns.set_index('Date', inplace=True)
        
        benchmark_returns = pd.DataFrame(index=spy_returns.index, columns=['Return'])
        
        for date in benchmark_returns.index:
            # 거래일 중 현재 날짜 이전의 날짜들 찾기
            valid_days = trading_days[trading_days <= date]
            
            if len(valid_days) == 0:
                continue
                
            if freq == 'week':
                # 주간 데이터의 경우 해당 날짜 이전 5 거래일 데이터 사용
                period_dates = valid_days.tail(5)
                period_data = df[df['date'].isin(period_dates)]
            else:
                # 해당 월/분기/연도의 거래일 찾기
                same_period_mask = (
                    (valid_days.dt.year == date.year) & 
                    (valid_days.dt.month == date.month)
                )
                period_dates = valid_days[same_period_mask]
                period_data = df[df['date'].isin(period_dates)]
            
            if not period_data.empty:
                # 해당 기간의 평균 수익률 계산
                period_returns = period_data.groupby('PERMNO')['Return'].apply(
                    lambda x: (1 + x).prod() - 1
                )
                valid_returns = period_returns[period_returns != 0]
                if not valid_returns.empty:
                    benchmark_returns.loc[date, 'Return'] = valid_returns.mean()
        
        # 결측치 처리
        benchmark_returns['Return'] = benchmark_returns['Return'].fillna(method='ffill')
        
        # 다음 기간의 수익률 추가
        benchmark_returns['nxt_freq_ewret'] = benchmark_returns['Return'].shift(-1)
        
        # parquet 파일로 저장
        benchmark_returns.reset_index().to_parquet(parquet_path, index=False)
        print(f"{freq} 벤치마크 수익률이 {parquet_path}에 저장되었습니다.")

def get_spy_freq_rets(freq):
    """S&P 500 주기별 수익률을 가져옵니다. 필요한 경우 파일을 생성합니다."""
    assert freq in ["week", "month", "quarter", 'year'], f"Invalid freq: {freq}"
    parquet_path = os.path.join(dcf.CACHE_DIR, f"spy_{freq}_ret.parquet")
    
    # 파일이 없으면 생성
    if not os.path.exists(parquet_path):
        create_spy_returns()
    
    # parquet 파일 로드
    spy = pd.read_parquet(parquet_path)
    spy.set_index('Date', inplace=True)
    return spy

def get_bench_freq_rets(freq):
    """벤치마크 주기별 수익률을 가져옵니다. 필요한 경우 파일을 생성합니다."""
    assert freq in ["week", "month", "quarter", "year"]
    parquet_path = os.path.join(dcf.CACHE_DIR, f"benchmark_{freq}_ret.parquet")
    
    # 파일이 없으면 생성
    if not os.path.exists(parquet_path):
        create_return_files()
    
    # parquet 파일 로드
    bench = pd.read_parquet(parquet_path)
    bench.set_index('Date', inplace=True)
    return bench

def create_return_files():
    """모든 수익률 파일을 생성합니다."""
    os.makedirs(dcf.CACHE_DIR, exist_ok=True)
    create_spy_returns()
    create_benchmark_returns()

def get_period_end_dates(period):
    """주기별 마지막 거래일을 가져옵니다."""
    assert period in ["week", "month", "quarter", "year"]
    
    try:
        # S&P 500 데이터에서 주기별 마지막 거래일 가져오기
        spy = get_spy_freq_rets(period)
        return spy.index
    except Exception as e:
        print(f"S&P 500 데이터에서 주기별 마지막 거래일을 가져오는 중 오류 발생: {e}")
        print("대체 방법으로 주기별 마지막 거래일을 생성합니다.")
        
        # 대체 방법: 2000년부터 2023년까지의 날짜 범위 생성
        start_date = pd.Timestamp('2000-01-01')
        end_date = pd.Timestamp('2023-12-31')
        
        if period == 'week':
            # 주별 마지막 거래일 (금요일)
            dates = pd.date_range(start=start_date, end=end_date, freq='W-FRI')
        elif period == 'month':
            # 월별 마지막 거래일
            dates = pd.date_range(start=start_date, end=end_date, freq='M')
        elif period == 'quarter':
            # 분기별 마지막 거래일
            dates = pd.date_range(start=start_date, end=end_date, freq='Q')
        elif period == 'year':
            # 연도별 마지막 거래일
            dates = pd.date_range(start=start_date, end=end_date, freq='Y')
        
        print(f"생성된 {period} 주기 마지막 거래일: {len(dates)}개")
        return dates

def processed_US_data():
    """미국 주식 데이터를 처리하고 feather 형식으로 저장합니다."""
    processed_us_data_path = op.join(dcf.PROCESSED_DATA_DIR, "us_ret.feather")
    
    # 이미 처리된 파일이 있으면 로드
    if op.exists(processed_us_data_path):
        print(f"이미 처리된 데이터를 로드합니다: {processed_us_data_path}")
        since = time.time()
        df = pd.read_feather(processed_us_data_path)
        df.set_index(["Date", "StockID"], inplace=True)
        df.sort_index(inplace=True)
        print(f"데이터 로드 완료: {(time.time() - since) / 60:.2f}분 소요")
        return df.copy()

    # 처리된 파일이 없으면 원본 데이터 로드 및 처리
    raw_us_data_path = op.join(dcf.FILTERED_DATA_DIR, "Data_filtered.csv")
    print(f"원본 데이터를 로드합니다: {raw_us_data_path}")
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
    )
    print(f"원본 데이터 로드 완료: {(time.time() - since) / 60:.2f}분 소요")
    
    # 데이터 처리
    df = process_raw_data_helper(df)
    
    # 처리된 데이터 저장
    print(f"처리된 데이터를 저장합니다: {processed_us_data_path}")
    os.makedirs(op.dirname(processed_us_data_path), exist_ok=True)
    df.reset_index().to_feather(processed_us_data_path)
    print(f"데이터 저장 완료")
    
    return df.copy()

def process_raw_data_helper(df):
    print("Processing raw data...")
    since = time.time()
    
    # 컬럼명 변경
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
    
    # 데이터 타입 변환 및 정리
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
    
    # 인덱스 설정
    df.set_index(["Date", "StockID"], inplace=True)
    df.sort_index(inplace=True)
    
    print(f"Basic processing completed in {(time.time() - since) / 60:.2f} min")
    
    # 수익률 계산
    df['Ret'] = df['Ret'].apply(lambda x: max(x, -0.9999))
    df["log_ret"] = np.log(1 + df.Ret)
    
    # 누적 로그 수익률 계산 - 이 부분은 계산 비용이 큼
    print("Calculating cumulative log returns...")
    since_cum = time.time()
    df["cum_log_ret"] = df.groupby("StockID")["log_ret"].cumsum(skipna=True)
    print(f"Cumulative log returns calculated in {(time.time() - since_cum) / 60:.2f} min")
    
    # EWMA 변동성 계산
    print("Calculating EWMA volatility...")
    since_ewma = time.time()
    df["EWMA_vol"] = df.groupby("StockID")["Ret"].transform(
        lambda x: (x**2).ewm(alpha=0.05).mean().shift(periods=1)
    )
    print(f"EWMA volatility calculated in {(time.time() - since_ewma) / 60:.2f} min")
    
    # 주기별 수익률 계산
    for freq in ["week", "month", "quarter", "year"]:
        print(f"Processing {freq} returns...")
        since_freq = time.time()
        
        # 주기별 마지막 날짜 가져오기
        period_end_dates = get_period_end_dates(freq)
        
        # 주기별 마지막 날짜에 해당하는 데이터만 필터링
        freq_df = df[df.index.get_level_values("Date").isin(period_end_dates)].copy()
        
        # 주기별 수익률 계산
        freq_df["freq_ret"] = freq_df.groupby("StockID")["cum_log_ret"].transform(
            lambda x: np.exp(x.shift(-1) - x) - 1
        )
        
        print(
            f"Freq: {freq}: {len(freq_df)}/{len(df)} period_end_dates from \
                        {period_end_dates[0]}, {period_end_dates[1]},  to {period_end_dates[-1]}"
        )
        
        # 원본 데이터프레임에 주기별 수익률 추가
        df[f"Ret_{freq}"] = freq_df["freq_ret"]
        num_nan = np.sum(pd.isna(df[f"Ret_{freq}"]))
        print(f"df Ret_{freq} {len(df) - num_nan}/{len(df)} not nan")
        print(f"{freq} returns processed in {(time.time() - since_freq) / 60:.2f} min")
    
    # 다양한 기간의 수익률 계산
    print("Calculating multi-day returns...")
    since_multi = time.time()
    
    # 병렬 처리를 위한 함수 정의
    def calculate_period_return(i):
        print(f"Calculating {i}d return")
        try:
            return df.groupby("StockID")["cum_log_ret"].transform(
                lambda x: np.exp(x.shift(-i) - x) - 1 if len(x) > i else pd.Series(np.nan, index=x.index)
            )
        except Exception as e:
            print(f"{i}일 수익률 계산 중 오류 발생: {e}")
            # 오류 발생 시 NaN으로 채운 시리즈 반환
            return pd.Series(np.nan, index=df.index)
    
    # 각 기간별로 수익률 계산
    for i in [5, 20, 60, 65, 180, 250, 260]:
        df[f"Ret_{i}d"] = calculate_period_return(i)
    
    print(f"Multi-day returns calculated in {(time.time() - since_multi) / 60:.2f} min")
    print(f"Total processing time: {(time.time() - since) / 60:.2f} min")
    
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
    """주기별 수익률 데이터를 가져옵니다. 필요한 경우 새로 생성합니다."""
    assert country == "USA"
    assert period in ["week", "month", "quarter"]
    period_ret_path = op.join(dcf.CACHE_DIR, f"us_{period}_ret.pq")
    
    # 파일이 없거나 delay 컬럼이 없는 경우 새로 생성
    need_update = False
    if not op.exists(period_ret_path):
        need_update = True
    else:
        try:
            # 기존 파일 로드
            period_ret = pd.read_parquet(period_ret_path)
            # delay 컬럼 확인
            expected_cols = [f'next_{period}_ret_{d}delay' for d in range(6)]
            if not all(col in period_ret.columns for col in expected_cols):
                need_update = True
        except Exception:
            need_update = True
    
    if need_update:
        print(f"{period_ret_path} needs update. Creating it now.")
        create_period_ret_file(period)
    
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
    """주기별 수익률 파일을 생성하고 delay 수익률도 함께 계산합니다."""
    print(f"{freq} 주기 수익률 파일을 생성합니다...")
    
    # 출력 파일 경로
    period_ret_path = op.join(dcf.CACHE_DIR, f"us_{freq}_ret.pq")
    
    # 이미 처리된 파일이 있으면 건너뛰기
    if op.exists(period_ret_path):
        try:
            # 기존 파일 로드하여 필요한 컬럼이 있는지 확인
            period_ret = pd.read_parquet(period_ret_path)
            expected_cols = [f'next_{freq}_ret_{d}delay' for d in range(6)]
            if all(col in period_ret.columns for col in expected_cols):
                print(f"{period_ret_path} 파일이 이미 존재하고 필요한 모든 컬럼을 포함합니다.")
                return
        except Exception as e:
            print(f"기존 파일 확인 중 오류 발생: {e}")
    
    # 처리된 주식 데이터 로드
    stock_path = op.join(dcf.PROCESSED_DATA_DIR, "us_ret.feather")
    if not op.exists(stock_path):
        print("처리된 주식 데이터가 없습니다. 데이터를 생성합니다...")
        processed_US_data()  # 필요한 경우 데이터 생성
    
    print(f"처리된 주식 데이터를 로드합니다: {stock_path}")
    since = time.time()
    stock = pd.read_feather(stock_path)
    stock = stock[['Date', 'StockID', 'MarketCap', 'Close', 'Ret']]
    print(f"데이터 로드 완료: {(time.time() - since) / 60:.2f}분 소요")
    
    us_freq_ret = stock.copy()
    
    # 기본 수익률 계산
    print("기본 수익률을 계산합니다...")
    us_freq_ret = calculate_next_ret(us_freq_ret, freq)
    
    # delay 수익률 계산 (0부터 시작하여 여러 delay 포함)
    print("지연 수익률을 계산합니다...")
    for delay in range(6):  # 0부터 5일까지의 delay
        print(f"지연 {delay}일 처리 중...")
        us_freq_ret = calculate_next_ret_delay(us_freq_ret, freq, delay)
    
    # 추가 수익률 계산
    print("추가 수익률을 계산합니다...")
    us_freq_ret = calculate_ret(us_freq_ret, 6, 20)
    us_freq_ret = calculate_ret(us_freq_ret, 6, 60)
    
    # 결과 저장
    print(f"결과를 저장합니다: {period_ret_path}")
    us_freq_ret.to_parquet(period_ret_path)
    print(f"{period_ret_path} 파일 생성 완료")
    
    # 데이터 확인
    delay_cols = [f'next_{freq}_ret_{d}delay' for d in range(6)]
    for col in delay_cols:
        nan_count = us_freq_ret[col].isna().sum()
        print(f"{col}: {nan_count}개의 NaN 값")

if __name__ == "__main__":
    create_return_files()
    for freq in ["week", "month", "quarter"]:
        create_period_ret_file(freq)