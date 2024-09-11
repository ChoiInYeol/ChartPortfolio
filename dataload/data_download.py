import os
import json
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


def stock_download(
    dic,
    start="2001-01-01",
    end="2021-11-30",
    len_data=5000,
    n_stock=50,
    download_dir="data/stocks/",
):
    os.makedirs(download_dir, exist_ok=True)
    count = 0
    stock_dict = {}
    for symbol in dic:
        symbol = symbol if symbol != "BRK.B" else "BRK-B"
        print(f"Downloading data for {symbol}...")
        
        data = get_stock_data(symbol, start, end)
        if data is None:
            print(f"Skipping {symbol} due to insufficient data.")
            continue

        if len(data) > len_data:
            data.to_csv(os.path.join(download_dir, f"{symbol}.csv"))
            stock_dict[symbol] = dic[symbol]
            count += 1
            print(f"Successfully downloaded {symbol}.")
        else:
            print(f"Failed to download sufficient data for {symbol}.")

        if count >= n_stock:
            break
    return stock_dict


if __name__ == "__main__":
    # Load configuration and symbols
    config = json.load(open("config/data_config.json", "r", encoding="utf8"))
    snp500 = pd.read_csv("data/snp500.csv")
    
    # Handle BRK.B symbol
    snp500.loc[snp500.Symbol == "BRK.B", "Symbol"] = "BRK-B"
    snp500 = {tup[2]: tup[1] for tup in snp500.values.tolist()}

    # Download stock data
    stock_pair = stock_download(
        snp500, len_data=config["LEN_DATA"], n_stock=config["N_STOCK"], download_dir="data/stocks/"
    )

    # Download S&P 500 index data
    sp500 = yf.download("^GSPC", config["START"], config["END"])
    sp500.to_csv("data/snp500_index.csv")

    # Save downloaded stock symbols to a JSON file
    json.dump(stock_pair, open("data/stock.json", "w", encoding="UTF-8"))
