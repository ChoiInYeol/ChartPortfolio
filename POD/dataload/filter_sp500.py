import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def filter_sp500_stocks(sp500_path, ticker_path, filtered_stock_path, output_path):
    """
    S&P500 종목들만 필터링하여 새로운 CSV 파일을 생성합니다.
    
    Args:
        sp500_path (str): S&P500 종목 리스트 파일 경로
        ticker_path (str): ticker_stockid.csv 파일 경로
        filtered_stock_path (str): 필터링할 주식 데이터 파일 경로
        output_path (str): 결과 저장 경로
    """
    # S&P500 종목 리스트 로드
    sp500_df = pd.read_csv(sp500_path, header=None, names=['TICKER'])
    sp500_tickers = set(sp500_df['TICKER'].dropna().values)
    logger.info(f"Loaded {len(sp500_tickers)} S&P500 tickers")
    
    # ticker_stockid 매핑 로드
    ticker_map_df = pd.read_csv(ticker_path)
    ticker_map = dict(zip(ticker_map_df['TICKER'], ticker_map_df['StockID']))
    logger.info(f"Loaded {len(ticker_map)} ticker mappings")
    
    # S&P500 종목의 StockID 찾기
    sp500_stockids = []
    for ticker in sp500_tickers:
        if ticker in ticker_map:
            sp500_stockids.append(ticker_map[ticker])
    
    logger.info(f"Found {len(sp500_stockids)} matching StockIDs")
    
    # 주식 데이터 로드 및 필터링
    stock_data = pd.read_csv(filtered_stock_path)
    filtered_data = stock_data[stock_data['PERMNO'].isin(sp500_stockids)]
    
    logger.info(f"Original data shape: {stock_data.shape}")
    logger.info(f"Filtered data shape: {filtered_data.shape}")
    
    # 결과 저장
    filtered_data.to_csv(output_path, index=False)
    logger.info(f"Saved filtered data to {output_path}")
    
    # 매칭된 종목 리스트 저장
    matched_tickers = pd.DataFrame({
        'TICKER': [t for t in sp500_tickers if t in ticker_map],
        'StockID': [ticker_map[t] for t in sp500_tickers if t in ticker_map]
    })
    matched_tickers.to_csv(output_path.replace('.csv', '_matched_tickers.csv'), index=False)
    logger.info(f"Saved matched tickers list")

if __name__ == "__main__":
    filter_sp500_stocks(
        sp500_path="POD/data/S&P 500 on 2018-01-01.csv",
        ticker_path="POD/data/ticker_stockid.csv",
        filtered_stock_path="POD/data/filtered_stock.csv",
        output_path="POD/data/sp500_filtered_stock.csv"
    ) 