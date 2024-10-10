import yfinance as yf
import numpy as np
import yaml
from tqdm import tqdm
import logging
import pandas as pd
import os

logging.basicConfig(filename='data_download.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_stock_data(symbol, start_date, end_date):
    """
    주어진 주식 심볼에 대한 데이터를 다운로드하고 전처리합니다.

    Args:
        symbol (str): 주식 심볼
        start_date (str): 데이터 시작 날짜
        end_date (str): 데이터 종료 날짜

    Returns:
        pd.DataFrame or None: 처리된 주식 데이터 또는 오류 발생 시 None
    """
    try:
        stock = yf.download(symbol, start=start_date, end=end_date)
        if stock.empty or len(stock) <= 252:  # 최소 1년치 데이터
            logging.warning(f"불충분한 데이터: {symbol}")
            return None
        
        stock = stock[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
        stock['Return'] = stock['Adj Close'].pct_change()
        stock['log_ret'] = np.log(1 + stock['Return'])
        stock['cum_log_ret'] = stock['log_ret'].cumsum()
        stock['EWMA_vol'] = stock['Return'].ewm(span=20).std()
        
        return stock
    except Exception as e:
        logging.error(f"{symbol} 다운로드 중 오류 발생: {e}")
        return None

def preprocess_stock_data(stock_dir="data/stocks/"):
    all_dates = pd.DatetimeIndex([])
    for file in os.listdir(stock_dir):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(stock_dir, file), index_col=0, parse_dates=True)
            all_dates = all_dates.union(df.index)

    start_date = all_dates.min()
    end_date = all_dates.max()
    date_range = pd.bdate_range(start=start_date, end=end_date)

    for file in tqdm(os.listdir(stock_dir), desc="전처리 중"):
        if file.endswith('.csv'):
            file_path = os.path.join(stock_dir, file)
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            df = df.reindex(date_range)
            # 결측치를 NaN으로 유지
            df.index.name = 'Date'  # 인덱스 이름을 'Date'로 설정
            
            # 칼럼 이름 변경
            df = df.rename(columns={'Volume': 'Vol', 'Return': 'Ret'})
            
            # 필요한 경우 추가 전처리 수행
            if 'Ret' not in df.columns:
                df['Ret'] = df['Adj Close'].pct_change()
            
            if 'log_ret' not in df.columns:
                df['log_ret'] = np.log(1 + df['Ret'])
            
            if 'cum_log_ret' not in df.columns:
                df['cum_log_ret'] = df['log_ret'].cumsum()
            
            if 'EWMA_vol' not in df.columns:
                df['EWMA_vol'] = df['Ret'].ewm(span=20).std()
            
            df.to_csv(file_path)
    
    logging.info("전처리 완료")

def download_stock_data(symbols, start_date, end_date, download_dir="data/stocks/"):
    os.makedirs(download_dir, exist_ok=True)
    stock_dict = {}
    
    for symbol in tqdm(symbols, desc="주식 데이터 다운로드 중"):
        symbol = symbol if symbol != "BRK.B" else "BRK-B"
        file_path = f"{download_dir}{symbol}.csv"
        
        if os.path.exists(file_path):
            logging.info(f"{symbol} 파일이 이미 존재합니다. 다운로드를 건너뜁니다.")
            stock_dict[symbol] = symbol
            continue
        
        stock = get_stock_data(symbol, start_date, end_date)
        
        if stock is not None:
            stock.to_csv(file_path)
            stock_dict[symbol] = symbol
            logging.info(f"{symbol} 다운로드 및 처리 완료")
        else:
            logging.warning(f"{symbol}에 대한 충분한 데이터를 다운로드하지 못했습니다.")
    
    return stock_dict

def recover_stock_data(stock_dir="data/stocks/"):
    for file in tqdm(os.listdir(stock_dir), desc="데이터 복구 중"):
        if file.endswith('.csv'):
            file_path = os.path.join(stock_dir, file)
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # -2를 NaN으로 변경
            df = df.replace(-2, np.nan)
            
            # 필요한 경우 추가 전처리 수행
            if 'Ret' in df.columns:
                df['Ret'] = df['Adj Close'].pct_change()
            
            if 'log_ret' in df.columns:
                df['log_ret'] = np.log(1 + df['Ret'])
            
            if 'cum_log_ret' in df.columns:
                df['cum_log_ret'] = df['log_ret'].cumsum()
            
            if 'EWMA_vol' in df.columns:
                df['EWMA_vol'] = df['Ret'].ewm(span=20).std()
            
            df.to_csv(file_path)
    
    logging.info("데이터 복구 완료")

if __name__ == "__main__":
    # YAML 설정 파일 로드
    with open("config/config.yaml", "r", encoding="utf8") as f:
        config = yaml.safe_load(f)
    
    recover_stock_data()
    
    # S&P 500 구성 종목 로드
    # snp500 = pd.read_csv("data/snp500.csv")
    # snp500.loc[snp500.Symbol == "BRK.B", "Symbol"] = "BRK-B"
    # symbols = snp500['Symbol'].tolist()
    
    # # 주식 데이터 다운로드
    # stock_dict = download_stock_data(symbols, config['START'], config['END'])
    
    # # S&P 500 지수 데이터 다운로드
    # sp500 = yf.download("^GSPC", config['START'], config['END'])
    # sp500.to_csv("data/snp500_index.csv")
    
    # # 다운로드된 주식 심볼을 YAML 파일로 저장
    # with open("data/stock.yaml", "w", encoding="UTF-8") as f:
    #     yaml.dump(stock_dict, f)
    
    # # 다운로드된 데이터 전처리
    # preprocess_stock_data()
    
    logging.info("주식 데이터 다운로드 및 전처리 프로세스 완료.")