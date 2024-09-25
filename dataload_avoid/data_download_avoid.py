import os
import yaml
import pandas as pd
import yfinance as yf

def get_stock_data(code, start, end):
    try:
        data = yf.download(code, start, end)
        if data.empty:
            print(f"No data found for {code}. Skipping.")
            return None
        
        # Drop rows where volume is too low
        data = data.drop(data[data.Volume < 10].index)
        if data.empty:
            print(f"Insufficient volume data for {code}. Skipping.")
            return None

        # Create business date range and reindex
        business_date = pd.bdate_range(data.index[0], data.index[-1])
        data = pd.DataFrame(data, index=business_date)
        data.index.name = "Date"
        
        # Interpolate missing values for 'Adj Close'
        data["Adj Close"] = data["Adj Close"].interpolate(method="linear")
        return data

    except Exception as e:
        print(f"Error downloading data for {code}: {e}")
        return None

def stock_download(tickers, start, end, download_dir):
    os.makedirs(download_dir, exist_ok=True)
    downloaded_tickers = []

    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        
        data = get_stock_data(ticker, start, end)
        if data is not None:
            data.to_csv(os.path.join(download_dir, f"{ticker}.csv"))
            downloaded_tickers.append(ticker)
            print(f"Successfully downloaded {ticker}.")
        else:
            print(f"Failed to download data for {ticker}.")

    return downloaded_tickers

if __name__ == "__main__":
    # Load SPY component status data
    sp500_status = pd.read_csv('sp500_ticker_status_1996_2024.csv')
    unique_tickers = sp500_status['ticker'].unique()

    # Download stock data
    start_date = "1996-01-02"
    end_date = "2024-07-08"
    download_dir = "data/stocks/"
    
    downloaded_tickers = stock_download(unique_tickers, start_date, end_date, download_dir)

    # Download S&P 500 index data
    sp500 = yf.download("^GSPC", start_date, end_date)
    sp500.to_csv("data/snp500_index.csv")

    # Save downloaded ticker list
    with open("data/downloaded_tickers.yaml", "w", encoding="UTF-8") as f:
        yaml.dump(downloaded_tickers, f)

    print(f"Downloaded data for {len(downloaded_tickers)} tickers.")