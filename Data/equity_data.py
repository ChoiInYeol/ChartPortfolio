import os
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from tqdm import tqdm
import yaml
from Data import dgp_config as dcf

logging.basicConfig(filename='equity_data.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def process_stock(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'Date' not in df.columns:
            logging.warning(f"'Date' column not found in {file_path}. Using first column as Date.")
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            df.index.name = 'Date'
        else:
            df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        
        stock_id = os.path.splitext(os.path.basename(file_path))[0]
        df['StockID'] = stock_id
        
        if 'Adj Close' not in df.columns:
            logging.warning(f"'Adj Close' column not found in {file_path}. Using 'Close' instead.")
            df['Adj Close'] = df['Close']
        
        df['log_ret'] = np.log1p(df['Adj Close'].pct_change())
        df['cum_log_ret'] = df['log_ret'].cumsum()
        
        if 'Return' not in df.columns:
            df['Return'] = df['Adj Close'].pct_change()
        
        df['EWMA_vol'] = df['Return'].ewm(span=20).std()
        
        logging.info(f"Successfully processed {stock_id}")
        return df
    except Exception as e:
        logging.warning(f"Error processing {file_path}: {e}")
        return None

def get_processed_US_data_by_year(year, stock_dir="data/stocks/"):
    all_data = []
    stock_files = [f for f in os.listdir(stock_dir) if f.endswith('.csv')]
    logging.info(f"Found {len(stock_files)} CSV files in {stock_dir}")
    
    for file in tqdm(stock_files, desc="Processing stocks"):
        file_path = os.path.join(stock_dir, file)
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        df['StockID'] = os.path.splitext(file)[0]
        all_data.append(df)
    
    if not all_data:
        logging.error("No stock data processed successfully")
        raise ValueError("No stock data processed successfully")
    
    df = pd.concat(all_data)
    df = df.reset_index()
    df.set_index(['Date', 'StockID'], inplace=True)
    df.sort_index(inplace=True)
    
    df = df[df.index.get_level_values("Date").year.isin([year, year - 1, year - 2])]
    
    for freq in ["week", "month", "quarter", "year"]:
        period_end_dates = get_period_end_dates(freq)
        freq_df = df[df.index.get_level_values("Date").isin(period_end_dates)].copy()
        freq_df[f"Ret_{freq}"] = freq_df.groupby("StockID")["cum_log_ret"].transform(
            lambda x: np.exp(x.shift(-1) - x) - 1
        )
        df[f"Ret_{freq}"] = freq_df[f"Ret_{freq}"]
    
    for i in [5, 20, 60, 65, 180, 250, 260]:
        df[f"Ret_{i}d"] = df.groupby("StockID")["cum_log_ret"].transform(
            lambda x: np.exp(x.shift(-i) - x) - 1
        )
    
    return df

def get_spy_data():
    spy_path = os.path.join(dcf.DATA_DIR, "snp500_index.csv")
    spy = pd.read_csv(spy_path, parse_dates=['Date'], index_col='Date')
    spy['Return'] = spy['Adj Close'].pct_change()
    return spy

def get_period_end_dates(period):
    assert period in ["week", "month", "quarter", "year"]
    spy = get_spy_data()
    if period == "week":
        return spy.resample('W').last().index
    elif period == "month":
        return spy.resample('ME').last().index
    elif period == "quarter":
        return spy.resample('QE').last().index
    else:  # year
        return spy.resample('YE').last().index

def get_spy_freq_rets(freq):
    assert freq in ["week", "month", "quarter"]
    spy = get_spy_data()
    if freq == "week":
        return spy.resample('WE').last()['Return']
    elif freq == "month":
        return spy.resample('ME').last()['Return']
    else:  # quarter
        return spy.resample('QE').last()['Return']

if __name__ == "__main__":
    with open("config/config.yaml", "r", encoding="utf8") as f:
        config = yaml.safe_load(f)
    
    year = config.get('YEAR', 2024)
    df = get_processed_US_data_by_year(year)
    
    output_path = os.path.join(dcf.PROCESSED_DATA_DIR, f"us_stock_data_{year}.feather")
    df.reset_index().to_feather(output_path)
    logging.info(f"Processed data for year {year} saved to {output_path}")