import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict
from tqdm import tqdm
import json
import pickle

def generate_candlestick_chart_images(
    return_df: pd.DataFrame,
    tickers: List[str],
    train_len: int,
    pred_len: int,
    tr_ratio: float,
    save_dir: str = "./data/charts",
    has_ma: bool = False,
    has_volume: bool = False,
    ma_period: int = 5,
    save_npz: bool = True,
    npz_file_name_prefix: str = "candlestick_data",
) -> Tuple[Dict, Dict, List, List]:
    size_dict = {5: (32, 15), 20: (64, 60), 60: (96, 180), 120: (128, 360)}

    split_index = int(len(return_df) * tr_ratio)
    train_data = return_df.iloc[:split_index]
    test_data = return_df.iloc[split_index:]

    train_images, train_labels, train_times = [], [], []
    test_images, test_labels, test_times = [], [], []

    for dataset, images, labels, times in [(train_data, train_images, train_labels, train_times),
                                           (test_data, test_images, test_labels, test_times)]:
        for ticker in tqdm(tickers, desc="Processing tickers"):
            ticker_df = pd.read_csv(f"./data/stocks/{ticker}.csv", index_col="Date", parse_dates=True)
            ticker_df = ticker_df[ticker_df.index.isin(dataset.index)]
            
            if has_ma:
                ticker_df['MA'] = ticker_df['Close'].rolling(window=ma_period).mean()
            
            for i in tqdm(range(len(ticker_df) - train_len - pred_len + 1), desc=f"Processing {ticker}"):
                sub_df = ticker_df.iloc[i:i+train_len]
                future_return = dataset[ticker].iloc[i+train_len+pred_len-1] - dataset[ticker].iloc[i+train_len-1]

                image = generate_single_candlestick_chart(
                    sub_df,
                    ticker,
                    train_len,
                    has_ma=has_ma,
                    has_volume=has_volume,
                    size=size_dict[train_len],
                )

                if image is not None:
                    images.append(np.array(image))
                    labels.append(1 if future_return > 0 else 0)
                    times.append(ticker_df.index[i+train_len+pred_len-1])

    train_npz_data = {'images': np.array(train_images), 'labels': np.array(train_labels)}
    test_npz_data = {'images': np.array(test_images), 'labels': np.array(test_labels)}

    if save_npz:
        np.savez_compressed(f"{save_dir}/train_{npz_file_name_prefix}.npz", **train_npz_data)
        np.savez_compressed(f"{save_dir}/test_{npz_file_name_prefix}.npz", **test_npz_data)
        print(f"Data saved to {save_dir}/train_{npz_file_name_prefix}.npz and {save_dir}/test_{npz_file_name_prefix}.npz")

    return train_npz_data, test_npz_data, train_times, test_times

def generate_single_candlestick_chart(
    df: pd.DataFrame,
    ticker: str,
    period: int,
    has_ma: bool = False,
    has_volume: bool = False,
    size: Tuple[int, int] = (64, 60),
) -> Image.Image:
    ohlc_height, image_width = size
    
    # 볼륨 차트 높이 조정
    volume_height = ohlc_height // 5 if has_volume else 0
    ohlc_height -= volume_height + (1 if has_volume else 0)
    image_height = ohlc_height + volume_height + (1 if has_volume else 0)

    candlestick_width = 3
    gap_width = (image_width - period * candlestick_width) // (period - 1)

    image = Image.new("L", (image_width, image_height), color=0)
    draw = ImageDraw.Draw(image)

    centers = np.arange(candlestick_width // 2, image_width, candlestick_width + gap_width)[:period]

    price_data = df[['Open', 'High', 'Low', 'Close']].iloc[-period:]
    min_price, max_price = price_data.min().min(), price_data.max().max()

    if min_price == max_price:
        print(f"Error for {ticker}: min_price equals max_price")
        return None

    def price_to_y(price):
        return int(ohlc_height * (1 - (price - min_price) / (max_price - min_price)))

    for i, (_, row) in enumerate(price_data.iterrows()):
        open_y, high_y, low_y, close_y = map(price_to_y, row[['Open', 'High', 'Low', 'Close']])
        center = int(centers[i])
        
        # 캔들스틱 그리기
        draw.point([center - 1, open_y], fill=255)  # 왼쪽 (Open)
        draw.line([(center, high_y), (center, low_y)], fill=255)  # 중앙 (High-Low)
        draw.point([center + 1, close_y], fill=255)  # 오른쪽 (Close)

    if has_ma:
        ma_values = df['MA'].iloc[-period:]
        ma_y = [price_to_y(val) for val in ma_values if not np.isnan(val)]
        ma_points = list(zip(centers[-len(ma_y):], ma_y))
        if len(ma_points) > 1:
            draw.line(ma_points, fill=255, width=1)

    if has_volume:
        volume_data = df['Volume'].iloc[-period:]
        max_volume = volume_data.max()

        if max_volume > 0:
            for i, volume in enumerate(volume_data):
                if not np.isnan(volume):
                    vol_bar_height = int(volume_height * (volume / max_volume))
                    vol_y_top = image_height - vol_bar_height
                    center = int(centers[i])
                    draw.line([center, vol_y_top, center, image_height - 1], fill=255)

    return image

if __name__ == "__main__":
    config = json.load(open("config/train_config.json", "r", encoding="utf8"))
    stock_data_dir = "./data/stocks/"
    save_dir = "./data/charts"
    os.makedirs(save_dir, exist_ok=True)

    # return_df.csv 파일 불러오기
    return_df = pd.read_csv("data/return_df.csv", index_col="Date", parse_dates=True)
    tickers = return_df.columns.tolist()

    result = generate_candlestick_chart_images(
        return_df,
        tickers=tickers,
        train_len=config["TRAIN_LEN"],
        pred_len=config["PRED_LEN"],
        tr_ratio=config["TRAIN_RATIO"],
        has_ma=True,
        ma_period=5,
        has_volume=True,
        save_dir=save_dir,
        save_npz=False,
        npz_file_name_prefix="candlestick_data",
    )

    train_npz_data, test_npz_data, train_times, test_times = result

    with open("data/date.pkl", "wb") as f:
        pickle.dump(test_times, f)

    print("데이터셋 검증:")
    print(f"Train images shape: {train_npz_data['images'].shape}")
    print(f"Train labels shape: {train_npz_data['labels'].shape}")
    print(f"Test images shape: {test_npz_data['images'].shape}")
    print(f"Test labels shape: {test_npz_data['labels'].shape}")
    print(f"Train times length: {len(train_times)}")
    print(f"Test times length: {len(test_times)}")
    
    # result를 저장
    with open("data/dataset_img.pkl", "wb") as f:
        pickle.dump(result, f)