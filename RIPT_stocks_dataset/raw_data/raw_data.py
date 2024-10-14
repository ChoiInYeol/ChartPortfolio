import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import logging
import glob
import warnings

# Set up logging
logging.basicConfig(filename='download_errors.log', level=logging.ERROR)

# Suppress future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# Read 'ticker.csv'
ticker_df = pd.read_csv('ticker.csv')

# Group tickers by exchange
grouped_by_exchange = ticker_df.groupby('Exchange')

# Function to download and adjust data for a single ticker
def download_ticker_data(ticker):
    try:
        # Download the full data available from Yahoo Finance
        stock = yf.Ticker(ticker)
        data = stock.history(period="max", auto_adjust=False)

        if data.empty:
            logging.error(f"No data for ticker {ticker}")
            return None

        # Calculate adjustment factor
        adj_factor = data['Adj Close'] / data['Close']
        # Adjust Open, High, Low, and Close
        data['Open'] *= adj_factor
        data['High'] *= adj_factor
        data['Low'] *= adj_factor
        data['Close'] = data['Adj Close']

        # Add TICKER and date columns
        data['TICKER'] = ticker
        data['date'] = data.index.strftime('%Y%m%d')
        return data

    except Exception as e:
        logging.error(f"Error downloading data for ticker {ticker}: {e}")
        return None

# Process each exchange
for exchange, group in grouped_by_exchange:
    tickers = group['Symbol'].unique()
    data_frames = []

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {executor.submit(download_ticker_data, ticker): ticker for ticker in tickers}
        for future in tqdm(as_completed(future_to_ticker), total=len(future_to_ticker),
                           desc=f"Downloading data for exchange {exchange}"):
            ticker = future_to_ticker[future]
            try:
                data = future.result()
                if data is not None:
                    data_frames.append(data)
            except Exception as e:
                logging.error(f"Error processing data for ticker {ticker}: {e}")

    # Merge and save data per exchange
    if data_frames:
        try:
            high_low_data = pd.concat(data_frames)
            high_low_data.to_parquet(f'./high_low_data_{exchange}.parquet', index=False)
        except ValueError as e:
            logging.error(f"Error processing data for exchange {exchange}: {e}")

# Combine all parquet files
files = glob.glob('./high_low_data_*.parquet')
files = [file for file in files if 'ETF' not in file]

data_frames = []
for file in files:
    temp = pd.read_parquet(file)
    exchange_name = file.split("_")[-1].split(".")[0]
    temp["exchange"] = exchange_name
    data_frames.append(temp)

# Combine data frames
high_low_data = pd.concat(data_frames)
high_low_data['date'] = pd.to_datetime(high_low_data['date'], format='%Y%m%d')

# Copy data for further processing
data = high_low_data.copy()

# Process data
data_copy = data.copy()
data_copy.sort_values(['TICKER', 'date'], inplace=True)
data_copy['Ret'] = data_copy.groupby('TICKER')['Close'].pct_change()
data_copy.dropna(subset=['Ret'], inplace=True)
data_copy['Shares'] = np.nan
data_copy['StockID'] = data_copy['TICKER'].astype('category').cat.codes
data_copy['Date'] = data_copy['date'].dt.strftime('%Y%m%d')
data_copy.drop(columns=['date'], inplace=True)
data_copy.drop(columns=['Close'], inplace=True)
data_copy.rename(columns={'Adj Close': 'Close', 'Volume': 'Vol'}, inplace=True)

# Save StockID and TICKER mapping
ticker_stockid = data_copy[['exchange', 'TICKER', 'StockID']].drop_duplicates()
ticker_stockid.to_csv('ticker_stockid.csv', index=False)

# Rename columns
columns = {
    'Date': 'date',
    'StockID': 'PERMNO',
    'Low': 'BIDLO',
    'High': 'ASKHI',
    'Close': 'PRC',
    'Vol': 'VOL',
    'Shares': 'SHROUT',
    'Open': 'OPENPRC',
    'Ret': 'RET'
}
data_copy.rename(columns=columns, inplace=True)
data_copy = data_copy[list(columns.values())]
data_copy.set_index('PERMNO', inplace=True)
data_copy = data_copy.round(3)

# Save final data to CSV
data_copy.to_csv('all_data.csv')
