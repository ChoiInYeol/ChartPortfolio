import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict
from tqdm import tqdm
import json
import pickle
import multiprocessing as mp

def process_ticker(args):
    ticker, window_data, future_data, train_len, has_ma, has_volume, ma_period, size_dict = args
    ticker_df = pd.read_csv(f"./data/stocks/{ticker}.csv", index_col="Date", parse_dates=True)
    ticker_df = ticker_df[ticker_df.index.isin(window_data.index)]
    
    if has_ma:
        ticker_df['MA'] = ticker_df['Close'].rolling(window=ma_period).mean()
    
    image = generate_single_candlestick_chart(
        ticker_df,
        ticker,
        train_len,
        has_ma=has_ma,
        has_volume=has_volume,
        size=size_dict[train_len],
    )

    if image is not None:
        future_return = future_data[ticker].iloc[-1] - window_data[ticker].iloc[-1]
        label = 1 if future_return > 0 else 0
        return np.array(image), label
    return None, None

def make_image_dataset(data, train_len, pred_len, n_stock, tickers, has_ma=False, has_volume=False, ma_period=5):
    times = []
    images = []
    labels = []
    size_dict = {5: (32, 15), 20: (64, 60), 60: (96, 180), 120: (128, 360)}

    with mp.Pool(processes=mp.cpu_count()) as pool:
        for i in tqdm(range(len(data) - train_len - pred_len + 1)):
            window_data = data.iloc[i:i+train_len]
            future_data = data.iloc[i+train_len:i+train_len+pred_len]
            times.append(future_data.index[-1])

            args = [(ticker, window_data, future_data, train_len, has_ma, has_volume, ma_period, size_dict) for ticker in tickers]
            results = pool.map(process_ticker, args)

            for image, label in results:
                if image is not None and label is not None:
                    images.append(image)
                    labels.append(label)

    return np.array(images), np.array(labels), times

def data_split_images(data, train_len, pred_len, tr_ratio, n_stock, tickers, has_ma=False, has_volume=False, ma_period=5):
    split_index = int(len(data) * tr_ratio)
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    train_images, train_labels, train_times = make_image_dataset(
        train_data, train_len, pred_len, n_stock, tickers, has_ma, has_volume, ma_period
    )
    test_images, test_labels, test_times = make_image_dataset(
        test_data, train_len, pred_len, n_stock, tickers, has_ma, has_volume, ma_period
    )

    x_tr = train_images.reshape(-1, n_stock, *train_images.shape[1:])
    y_tr = train_labels.reshape(-1, n_stock)
    x_te = test_images.reshape(-1, n_stock, *test_images.shape[1:])
    y_te = test_labels.reshape(-1, n_stock)

    times_tr = np.unique(train_times).tolist()
    times_te = np.unique(test_times).tolist()

    return x_tr, y_tr, x_te, y_te, times_tr, times_te

def generate_single_candlestick_chart(
    df: pd.DataFrame,
    ticker: str,
    period: int,
    has_ma: bool = False,
    has_volume: bool = False,
    size: Tuple[int, int] = (64, 60),
) -> Image.Image:
    ohlc_height, image_width = size
    
    volume_height = ohlc_height // 5 if has_volume else 0
    ohlc_height -= volume_height + (1 if has_volume else 0)
    image_height = ohlc_height + volume_height + (1 if has_volume else 0)

    candlestick_width = 3
    gap_width = (image_width - period * candlestick_width) // (period - 1)

    image = Image.new("L", (image_width, image_height), color=0)
    draw = ImageDraw.Draw(image)

    centers = np.arange(candlestick_width // 2, image_width, candlestick_width + gap_width)[:period]

    price_data = df[['Open', 'High', 'Low', 'Close']].iloc[-period:]
    
    # NaN이 아닌 값들만 사용하여 min, max 계산
    valid_prices = price_data.values[~np.isnan(price_data.values)]
    if len(valid_prices) == 0:
        print(f"Error for {ticker}: All data is NaN")
        return None

    min_price, max_price = np.min(valid_prices), np.max(valid_prices)

    if min_price == max_price:
        print(f"Error for {ticker}: min_price equals max_price")
        return None

    def price_to_y(price):
        if np.isnan(price):
            return None
        return int(ohlc_height * (1 - (price - min_price) / (max_price - min_price)))

    for i, (_, row) in enumerate(price_data.iterrows()):
        open_y, high_y, low_y, close_y = map(price_to_y, row[['Open', 'High', 'Low', 'Close']])
        center = int(centers[i])
        
        # NaN 값이 있는 경우 해당 캔들스틱을 그리지 않음
        if None in (open_y, high_y, low_y, close_y):
            continue
        
        # 캔들스틱 그리기
        draw.point([center - 1, open_y], fill=255)  # 왼쪽 (Open)
        draw.line([(center, high_y), (center, low_y)], fill=255)  # 중앙 (High-Low)
        draw.point([center + 1, close_y], fill=255)  # 오른쪽 (Close)

    if has_ma:
        ma_values = df['MA'].iloc[-period:]
        ma_y = [price_to_y(val) for val in ma_values]
        ma_points = [(centers[i], y) for i, y in enumerate(ma_y) if y is not None]
        if len(ma_points) > 1:
            draw.line(ma_points, fill=255, width=1)

    if has_volume:
        volume_data = df['Volume'].iloc[-period:]
        max_volume = np.nanmax(volume_data)

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

    x_tr, y_tr, x_te, y_te, times_tr, times_te = data_split_images(
        return_df,
        config["TRAIN_LEN"],
        config["PRED_LEN"],
        config["TRAIN_RATIO"],
        len(tickers),
        tickers,
        has_ma=True,
        has_volume=True,
        ma_period=5
    )

    train_npz_data = {'images': x_tr, 'labels': y_tr}
    test_npz_data = {'images': x_te, 'labels': y_te}

    np.savez_compressed(f"{save_dir}/train_candlestick_data.npz", **train_npz_data)
    np.savez_compressed(f"{save_dir}/test_candlestick_data.npz", **test_npz_data)

    with open("data/date.pkl", "wb") as f:
        pickle.dump({'train': times_tr, 'test': times_te}, f)

    print("데이터셋 검증:")
    print(f"Train images shape: {x_tr.shape}")
    print(f"Train labels shape: {y_tr.shape}")
    print(f"Test images shape: {x_te.shape}")
    print(f"Test labels shape: {y_te.shape}")
    print(f"Train times length: {len(times_tr)}")
    print(f"Test times length: {len(times_te)}")
    
    # result를 저장
    result = (train_npz_data, test_npz_data, times_tr, times_te)
    with open("data/dataset_img.pkl", "wb") as f:
        pickle.dump(result, f)