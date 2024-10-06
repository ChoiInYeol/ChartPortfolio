import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from typing import Tuple, List, Dict, Generator
from tqdm import tqdm
import yaml
import pickle
import multiprocessing as mp
import logging

# 로깅 설정
logging.basicConfig(filename='make_image_dataset.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def process_single_ticker(args):
    ticker, data, train_len, pred_len, has_ma, has_volume, ma_period, size_dict = args
    ticker_df = pd.read_csv(f"./data/stocks/{ticker}.csv", index_col=0, parse_dates=True)
    ticker_df.index.name = 'Date'
    ticker_df = ticker_df.replace(-1, np.nan)
    
    if has_ma:
        ticker_df['MA'] = ticker_df['Close'].rolling(window=ma_period).mean()
    
    images = []
    labels = []
    times = []
    
    for i in range(len(data) - train_len - pred_len + 1):
        window_data = data.iloc[i:i+train_len]
        future_data = data.iloc[i+train_len:i+train_len+pred_len]
        
        ticker_window = ticker_df[ticker_df.index.isin(window_data.index)]
        
        image = generate_single_candlestick_chart(
            ticker_window,
            train_len,
            has_ma=has_ma,
            has_volume=has_volume,
            size=size_dict[train_len],
        )
        
        if image is not None and not is_empty_image(image):
            future_return = future_data[ticker].iloc[-1] - window_data[ticker].iloc[-1]
            label = 1 if future_return > 0 else 0
            images.append(np.array(image))
            labels.append(label)
            times.append(future_data.index[-1])
    
    return ticker, images, labels, times

def make_image_dataset(data, train_len, pred_len, tickers, has_ma=False, has_volume=False, ma_period=5):
    size_dict = {5: (32, 15), 20: (64, 60), 60: (96, 180), 120: (128, 360)}
    
    args_list = [(ticker, data, train_len, pred_len, has_ma, has_volume, ma_period, size_dict) for ticker in tickers]
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(process_single_ticker_and_save, args_list), total=len(args_list), desc="Processing tickers"):
            pass  # 각 프로세스에서 결과를 저장하므로 메인 프로세스에서는 아무것도 하지 않습니다.

def process_single_ticker_and_save(args):
    ticker, data, train_len, pred_len, has_ma, has_volume, ma_period, size_dict = args
    ticker, images, labels, times = process_single_ticker(args)
    if images and labels and times:
        save_intermediate_result(ticker, images, labels, times)

def save_intermediate_result(ticker: str, images: List, labels: List, times: List):
    save_dir = "./data/charts/intermediate"
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(f"{save_dir}/{ticker}_images.npy", np.array(images))
    np.save(f"{save_dir}/{ticker}_labels.npy", np.array(labels))
    with open(f"{save_dir}/{ticker}_times.pkl", "wb") as f:
        pickle.dump(times, f)


def load_intermediate_results(tickers: List[str], chunk_size: int = 10) -> Generator[Dict[str, Tuple[np.ndarray, np.ndarray, List]], None, None]:
    save_dir = "./data/charts/intermediate"
    
    for i in range(0, len(tickers), chunk_size):
        chunk_tickers = tickers[i:i+chunk_size]
        result = {}
        
        for ticker in chunk_tickers:
            images = np.load(f"{save_dir}/{ticker}_images.npy")
            labels = np.load(f"{save_dir}/{ticker}_labels.npy")
            with open(f"{save_dir}/{ticker}_times.pkl", "rb") as f:
                times = pickle.load(f)
            result[ticker] = (images, labels, times)
        
        yield result

def data_split_images(ticker_data_generator, train_ratio: float, val_ratio: float):
    train_images, train_labels, val_images, val_labels, test_images, test_labels = [], [], [], [], [], []
    train_times, val_times, test_times = [], [], []
    
    for ticker_chunk in ticker_data_generator:
        for ticker, (images, labels, times) in ticker_chunk.items():
            total_len = len(images)
            train_end = int(total_len * train_ratio)
            val_end = int(total_len * (train_ratio + val_ratio))
            
            train_images.extend(images[:train_end])
            train_labels.extend(labels[:train_end])
            train_times.extend(times[:train_end])
            
            val_images.extend(images[train_end:val_end])
            val_labels.extend(labels[train_end:val_end])
            val_times.extend(times[train_end:val_end])
            
            test_images.extend(images[val_end:])
            test_labels.extend(labels[val_end:])
            test_times.extend(times[val_end:])
    
    return (np.array(train_images), np.array(train_labels), 
            np.array(val_images), np.array(val_labels), 
            np.array(test_images), np.array(test_labels),
            train_times, val_times, test_times)

def process_ticker(args):
    ticker, window_data, future_data, train_len, has_ma, has_volume, ma_period, size_dict = args
    ticker_df = pd.read_csv(f"./data/stocks/{ticker}.csv", index_col=0, parse_dates=True)
    ticker_df.index.name = 'Date'  # 인덱스 이름을 'Date'로 설정
    ticker_df = ticker_df[ticker_df.index.isin(window_data.index)]
    
    # -1 값을 NaN으로 변경
    ticker_df = ticker_df.replace(-1, np.nan)
    
    if has_ma:
        ticker_df['MA'] = ticker_df['Close'].rolling(window=ma_period).mean()
    
    image = generate_single_candlestick_chart(
        ticker_df,
        train_len,
        has_ma=has_ma,
        has_volume=has_volume,
        size=size_dict[train_len],
    )

    if image is not None and not is_empty_image(image):
        future_return = future_data[ticker].iloc[-1] - window_data[ticker].iloc[-1]
        label = 1 if future_return > 0 else 0
        return np.array(image), label
    return None, None

def is_empty_image(image: Image.Image) -> bool:
    """이미지가 완전히 비어있는지 (모두 검은색인지) 확인합니다."""
    return not image.getbbox()

def generate_single_candlestick_chart(
    df: pd.DataFrame,
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
        return image

    min_price, max_price = np.min(valid_prices), np.max(valid_prices)

    if min_price == max_price:
        return image

    def price_to_y(price):
        if np.isnan(price):
            return None
        return int(ohlc_height * (1 - (price - min_price) / (max_price - min_price)))

    for i, (_, row) in enumerate(price_data.iterrows()):
        open_y, high_y, low_y, close_y = map(price_to_y, row[['Open', 'High', 'Low', 'Close']])
        center = int(centers[i])
        
        # 하나라도 유효한 값이 있으면 캔들스틱을 그림
        if any(y is not None for y in (open_y, high_y, low_y, close_y)):
            if high_y is not None and low_y is not None:
                draw.line([(center, high_y), (center, low_y)], fill=255)  # 중앙 (High-Low)
            if open_y is not None:
                draw.point([center - 1, open_y], fill=255)  # 왼쪽 (Open)
            if close_y is not None:
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
    # YAML 파일에서 설정 읽기
    with open("config/config.yaml", "r", encoding="utf8") as file:
        config = yaml.safe_load(file)

    save_dir = "./data/charts"
    intermediate_dir = "./data/charts/intermediate"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)

    # return_df.csv 파일 불러오기
    return_df = pd.read_csv("data/return_df.csv", index_col="Date", parse_dates=True)
    tickers = return_df.columns.tolist()

    # 중간 결과 파일이 존재하는지 확인하고 없는 티커만 리스트로 만듭니다
    missing_tickers = [ticker for ticker in tickers if not os.path.exists(f"{intermediate_dir}/{ticker}_images.npy")]

    if missing_tickers:
        logging.info(f"다음 티커의 중간 결과 파일이 없습니다: {missing_tickers}")
        logging.info("누락된 이미지 데이터를 생성합니다.")
        # 누락된 티커에 대해서만 이미지 생성 및 즉시 저장
        make_image_dataset(
            return_df,
            config["TRAIN_LEN"],
            config["PRED_LEN"],
            missing_tickers,
            has_ma=True,
            has_volume=True,
            ma_period=5
        )
    else:
        logging.info("모든 중간 결과 파일이 존재합니다. 이미지 생성 과정을 건너뜁니다.")

    # 중간 결과 로드
    logging.info("중간 결과 로드")
    ticker_data_generator = load_intermediate_results(tickers)

    # 데이터 분할
    logging.info("데이터 분할")
    x_tr, y_tr, x_val, y_val, x_te, y_te, times_tr, times_val, times_te = data_split_images(
        ticker_data_generator,
        config["TRAIN_RATIO"],
        config["VAL_RATIO"]
    )

    train_npz_data = {'images': x_tr, 'labels': y_tr}
    val_npz_data = {'images': x_val, 'labels': y_val}
    test_npz_data = {'images': x_te, 'labels': y_te}
    
    logging.info("데이터 저장")
    np.savez_compressed(f"{save_dir}/train_candlestick_data.npz", **train_npz_data)
    np.savez_compressed(f"{save_dir}/val_candlestick_data.npz", **val_npz_data)
    np.savez_compressed(f"{save_dir}/test_candlestick_data.npz", **test_npz_data)

    with open("data/date.pkl", "wb") as f:
        pickle.dump({'train': times_tr, 'val': times_val, 'test': times_te}, f)

    logging.info("데이터셋 검증:")
    logging.info(f"Train images shape: {x_tr.shape}")
    logging.info(f"Train labels shape: {y_tr.shape}")
    logging.info(f"Validation images shape: {x_val.shape}")
    logging.info(f"Validation labels shape: {y_val.shape}")
    logging.info(f"Test images shape: {x_te.shape}")
    logging.info(f"Test labels shape: {y_te.shape}")
    logging.info(f"Train times length: {len(times_tr)}")
    logging.info(f"Validation times length: {len(times_val)}")
    logging.info(f"Test times length: {len(times_te)}")
    
    # result를 저장
    result = (train_npz_data, val_npz_data, test_npz_data, times_tr, times_val, times_te)
    with open("data/dataset_img.pkl", "wb") as f:
        pickle.dump(result, f)

    logging.info("이미지 데이터셋 생성 완료")