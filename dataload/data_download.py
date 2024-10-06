import os
import yaml
import pandas as pd
import yfinance as yf
import logging

# 로깅 설정
logging.basicConfig(filename='stock_download.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_stock_data(code, start, end):
    """
    주어진 주식 코드에 대한 데이터를 다운로드하고 전처리합니다.

    Args:
        code (str): 주식 심볼
        start (str): 데이터 시작 날짜
        end (str): 데이터 종료 날짜

    Returns:
        pd.DataFrame or None: 처리된 주식 데이터 또는 오류 발생 시 None
    """
    try:
        data = yf.download(code, start, end)
        if data.empty:
            logging.warning(f"No data found for {code}. Skipping.")
            return None
        
        # Drop rows where volume is too low
        data = data.drop(data[data.Volume < 10].index)
        if data.empty:
            logging.warning(f"Insufficient volume data for {code}. Skipping.")
            return None

        # Create business date range and reindex
        business_date = pd.bdate_range(data.index[0], data.index[-1])
        data = pd.DataFrame(data, index=business_date)
        data.index.name = "Date"
        
        # Interpolate missing values for 'Adj Close'
        data["Adj Close"] = data["Adj Close"].interpolate(method="linear")
        return data

    except Exception as e:
        logging.error(f"Error downloading data for {code}: {e}")
        return None

def stock_download(
    dic,
    start="2001-01-01",
    end="2024-09-01",
    len_data=5000,
    download_dir="data/stocks/",
):
    """
    주어진 주식 목록에 대한 데이터를 다운로드하고 저장합니다.

    Args:
        dic (dict): 주식 심볼과 이름을 포함하는 딕셔너리
        start (str): 데이터 시작 날짜
        end (str): 데이터 종료 날짜
        len_data (int): 최소 데이터 길이
        download_dir (str): 데이터를 저장할 디렉토리 경로

    Returns:
        dict: 성공적으로 다운로드된 주식의 딕셔너리
    """
    os.makedirs(download_dir, exist_ok=True)
    stock_dict = {}
    for symbol in dic:
        symbol = symbol if symbol != "BRK.B" else "BRK-B"
        logging.info(f"Downloading data for {symbol}...")
        
        data = get_stock_data(symbol, start, end)
        if data is None:
            logging.warning(f"Skipping {symbol} due to insufficient data.")
            continue

        if len(data) > len_data:
            data.to_csv(os.path.join(download_dir, f"{symbol}.csv"))
            stock_dict[symbol] = dic[symbol]
            logging.info(f"Successfully downloaded {symbol}.")
        else:
            logging.warning(f"Failed to download sufficient data for {symbol}.")

    return stock_dict

if __name__ == "__main__":
    # YAML 설정 파일 로드
    with open("config/config.yaml", "r", encoding="utf8") as f:
        config = yaml.safe_load(f)
    
    snp500 = pd.read_csv("data/snp500.csv")
    
    # BRK.B 심볼 처리
    snp500.loc[snp500.Symbol == "BRK.B", "Symbol"] = "BRK-B"
    snp500 = {tup[2]: tup[1] for tup in snp500.values.tolist()}
    
    # 주식 데이터 다운로드
    stock_pair = stock_download(
        snp500, len_data=config["LEN_DATA"], download_dir="data/stocks/"
    )

    # S&P 500 지수 데이터 다운로드
    sp500 = yf.download("^GSPC", config["START"], config["END"])
    sp500.to_csv("data/snp500_index.csv")

    # 다운로드된 주식 심볼을 YAML 파일로 저장
    with open("data/stock.yaml", "w", encoding="UTF-8") as f:
        yaml.dump(stock_pair, f)

    logging.info("Stock download process completed.")