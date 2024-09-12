import os
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from make_image import generate_candlestick_chart_images, generate_single_candlestick_chart

def generate_sample_images(
    df: pd.DataFrame, 
    tickers: list, 
    train_len: int, 
    save_dir: str, 
    size_dict: dict, 
    num_samples: int = 5,
    has_ma: bool = False, 
    has_volume: bool = False, 
    ma_period: int = 5
) -> list:
    sample_results = []
    selected_tickers = np.random.choice(tickers, min(num_samples, len(tickers)), replace=False)
    
    for ticker in tqdm(selected_tickers, desc="Generating samples"):
        ticker_df = df[df['Ticker'] == ticker].sort_index()
        
        if has_ma:
            ticker_df['MA'] = ticker_df['Close'].rolling(window=ma_period).mean()
        
        if len(ticker_df) >= train_len + 5:  # 20개의 연속적인 샘플을 위한 충분한 데이터 확인
            start_idx = np.random.randint(0, len(ticker_df) - train_len - 5 + 1)
            
            for i in range(5):
                sub_df = ticker_df.iloc[start_idx + i : start_idx + i + train_len]
                
                image = generate_single_candlestick_chart(
                    sub_df,
                    train_len,
                    has_ma=has_ma,
                    has_volume=has_volume,
                    size=size_dict[train_len],
                    period=train_len
                )
                
                if image is not None:
                    file_name = f"sample_{ticker}_{train_len}D_{i+1}.png"
                    save_path = os.path.join(save_dir, file_name)
                    image.save(save_path, format='PNG')
                    
                    log_return = sub_df['Log_Return'].iloc[-1]
                    label = 1 if log_return > 0 else 0
                    
                    sample_results.append({
                        'Ticker': ticker,
                        'Period': train_len,
                        'Log_Return': log_return,
                        'Image_Path': save_path,
                        'Label': label,
                        'Sample_Number': i + 1
                    })
    
    return sample_results

def load_stock_data(stock_data_dir: str) -> pd.DataFrame:
    print("Loading stock data...")
    tickers = [f.split(".")[0] for f in os.listdir(stock_data_dir) if f.endswith(".csv")]
    dataframes = []
    
    for ticker in tqdm(tickers, desc="Loading stock data"):
        df = pd.read_csv(os.path.join(stock_data_dir, f"{ticker}.csv"), index_col="Date", parse_dates=True)
        df['Ticker'] = ticker
        df['Log_Return'] = np.log(df['Adj Close']) - np.log(df['Adj Close'].shift(1))
        dataframes.append(df)
    
    full_df = pd.concat(dataframes)
    full_df.dropna(inplace=True)
    
    return full_df, tickers

def test_generate_candlestick_chart_images(full_df: pd.DataFrame, tickers: list, config: dict):
    print("\n테스트: generate_candlestick_chart_images 함수 호출")
    test_ticker = np.random.choice(tickers)
    test_df = full_df[full_df['Ticker'] == test_ticker]
    
    test_result = generate_candlestick_chart_images(
        test_df,
        tickers=[test_ticker],
        train_len=config["TRAIN_LEN"],
        pred_len=config["PRED_LEN"],
        tr_ratio=config["TRAIN_RATIO"],
        has_ma=True,
        has_volume=True,
        ma_period=5,
        save_npz=True
    )
    
    if test_result:
        print(f"generate_candlestick_chart_images 함수 테스트 성공 (테스트 티커: {test_ticker})")
        train_npz_data, test_npz_data, train_times, test_times = test_result
        print(f"Train images shape: {train_npz_data['images'].shape}")
        print(f"Test images shape: {test_npz_data['images'].shape}")
    else:
        print(f"generate_candlestick_chart_images 함수 테스트 실패 (테스트 티커: {test_ticker})")

if __name__ == "__main__":
    config = json.load(open("config/train_config.json", "r", encoding="utf8"))
    stock_data_dir = "./data/stocks/"
    save_dir = "./data/sample_charts"
    os.makedirs(save_dir, exist_ok=True)

    full_df, tickers = load_stock_data(stock_data_dir)

    size_dict = {5: (32, 15), 20: (64, 60), 60: (96, 180), 120: (128, 360)}

    print("Generating sample images...")
    sample_results = generate_sample_images(
        full_df,
        tickers=tickers,
        train_len=config["TRAIN_LEN"],
        save_dir=save_dir,
        size_dict=size_dict,
        num_samples=config.get("NUM_SAMPLES", 2),
        has_ma=True,
        has_volume=True,
        ma_period=5
        
    )

    sample_df = pd.DataFrame(sample_results)
    sample_df.to_csv("data/sample_metadata.csv", index=False)
    print("Sample metadata saved to data/sample_metadata.csv")

    print("샘플 이미지 생성 완료")
    print(f"생성된 샘플 이미지 수: {len(sample_results)}")

    test_generate_candlestick_chart_images(full_df, tickers, config)