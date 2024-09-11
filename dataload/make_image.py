import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict
from tqdm import tqdm

def generate_candlestick_chart_images(
    df: pd.DataFrame,
    tickers: List[str],
    periods: List[int],
    save_dir: str = "./data/charts",
    has_ma: bool = False,
    has_volume: bool = False,
    ma_period: int = 5,
    save_npz: bool = False,
    npz_file_name_prefix: str = "candlestick_data",
    only_sample: bool = False,
    num_samples: int = 5
) -> pd.DataFrame:
    
    size_dict = {5: (32, 15), 20: (64, 60), 60: (96, 180), 120: (128, 360)}

    if only_sample:
        sample_results = generate_sample_images(df, tickers, periods, save_dir, size_dict, num_samples, has_ma, has_volume, ma_period)
        return pd.DataFrame(sample_results)
        
    npz_data = {period: {ticker: {'images': [], 'labels': [], 'meta_data': []} for ticker in tickers} for period in periods}
    
    for ticker in tqdm(tickers, desc="Processing tickers"):
        ticker_df = df[df['Ticker'] == ticker].sort_index()
        
        # MA 미리 계산
        if has_ma:
            ticker_df['MA'] = ticker_df['Close'].rolling(window=ma_period).mean()
            
        for period in periods:
            for start_idx in tqdm(range(0, len(ticker_df) - period + 1, period), desc=f"Processing period {period}"):
                sub_df = ticker_df.iloc[start_idx:start_idx + period]

                start_date = sub_df.index[0].strftime('%Y-%m-%d')
                end_date = sub_df.index[-1].strftime('%Y-%m-%d')
                image = generate_single_candlestick_chart(
                    sub_df,
                    ticker,
                    start_date,
                    end_date,
                    period,
                    has_ma=has_ma,
                    has_volume=has_volume,
                    size=size_dict[period],
                    save_dir=save_dir if not save_npz else None
                )

                if image is not None:
                    log_return = sub_df['Log_Return'].iloc[-1]
                    label = 1 if log_return > 0 else 0

                    meta_data_item = {
                        'Ticker': ticker,
                        'start': start_date,
                        'end': end_date,
                        'Period': period,
                        'Log_Return': log_return
                    }

                    if save_npz:
                        npz_data[period][ticker]['images'].append(np.array(image))
                        npz_data[period][ticker]['labels'].append(label)
                        npz_data[period][ticker]['meta_data'].append(meta_data_item)

    if save_npz:
        for period in periods:
            for ticker in tickers:
                if npz_data[period][ticker]['images']:
                    save_npz_file(
                        npz_data[period][ticker]['images'],
                        npz_data[period][ticker]['labels'],
                        npz_data[period][ticker]['meta_data'],
                        ticker,
                        period,
                        npz_file_name_prefix
                    )

    results_df = pd.DataFrame(npz_data)
    return results_df

def generate_sample_images(df: pd.DataFrame, tickers: List[str], periods: List[int], 
                           save_dir: str, size_dict: Dict[int, Tuple[int, int]], num_samples: int = 5,
                           has_ma: bool = False,
                           has_volume: bool = False,
                           ma_period: int = 5) -> List[Dict]:
    sample_results = []
    for period in periods:
        period_df = df.groupby('Ticker').last().reset_index()
        samples = period_df.sample(min(num_samples, len(period_df)))
        
        for _, row in samples.iterrows():
            ticker = row['Ticker']
            ticker_df = df[df['Ticker'] == ticker].sort_index()
            
            # MA 미리 계산
            if has_ma:
                ticker_df['MA'] = ticker_df['Close'].rolling(window=ma_period).mean()
            
            if len(ticker_df) >= period:
                sub_df = ticker_df.iloc[-period:]
                start_date = sub_df.index[0].strftime('%Y-%m-%d')
                end_date = sub_df.index[-1].strftime('%Y-%m-%d')
                
                image = generate_single_candlestick_chart(
                    sub_df,
                    ticker,
                    start_date,
                    end_date,
                    period,
                    has_ma=has_ma,
                    has_volume=has_volume,
                    ma_period=ma_period,
                    size=size_dict[period],
                    save_dir=save_dir
                )
                
                if image is not None:
                    log_return = sub_df['Log_Return'].iloc[-1]
                    label = 1 if log_return > 0 else 0
                    
                    sample_results.append({
                        'Ticker': ticker,
                        'start': start_date,
                        'end': end_date,
                        'Period': period,
                        'Log_Return': log_return,
                        'Image_Path': f"{save_dir}/sample_{ticker}_{start_date}_{end_date}_{period}D.png",
                        'Label': label
                    })
    
    return sample_results


def generate_single_candlestick_chart(
    df: pd.DataFrame,
    ticker: str,
    start_date: str,
    end_date: str,
    period: int,
    has_ma: bool = False,
    has_volume: bool = False,
    size: Tuple[int, int] = (64, 60),
    save_dir: str = "./data/charts"
) -> Image.Image:
    try:
        ohlc_height, image_width = size
        
        # 볼륨 차트 높이 조정
        if has_volume:
            volume_height = ohlc_height // 5
            ohlc_height = ohlc_height - volume_height - 1  # 1픽셀 간격
        else:
            volume_height = 0
        
        image_height = ohlc_height + volume_height + (1 if has_volume else 0)

        candlestick_width = 3
        gap_width = (image_width - period * candlestick_width) // (period - 1)

        image = Image.new("L", (image_width, image_height), color=0)
        draw = ImageDraw.Draw(image)

        centers = np.arange(candlestick_width // 2, image_width, candlestick_width + gap_width)[:period]

        price_data = df[['Open', 'High', 'Low', 'Close']].iloc[-period:]
        min_price = price_data.min().min()
        max_price = price_data.max().max()

        def price_to_y(price):
            return int(ohlc_height * (1 - (price - min_price) / (max_price - min_price)))

        for i, (_, row) in enumerate(price_data.iterrows()):
            open_y = price_to_y(row['Open'])
            high_y = price_to_y(row['High'])
            low_y = price_to_y(row['Low'])
            close_y = price_to_y(row['Close'])

            center = int(centers[i])
            # 왼쪽 (Open)
            draw.point([center - 1, open_y], fill=255)
            
            # 중앙 (High-Low)
            draw.line([(center, high_y), (center, low_y)], fill=255)
            
            # 오른쪽 (Close)
            draw.point([center + 1, close_y], fill=255)

        if has_ma:
            ma_values = df['MA'].iloc[-period:]  # 미리 계산된 MA 사용
            ma_y = [price_to_y(val) for val in ma_values if not np.isnan(val)]
            ma_points = list(zip(centers[-len(ma_y):], ma_y))
            if len(ma_points) > 1:
                draw.line(ma_points, fill=255, width=1)

        if has_volume:
            volume_data = df['Volume'].iloc[-period:]
            max_volume = volume_data.max()

            for i, volume in enumerate(volume_data):
                if np.isnan(volume) or max_volume == 0:
                    continue
                vol_bar_height = int(volume_height * (volume / max_volume))
                vol_y_top = image_height - vol_bar_height
                center = int(centers[i])
                draw.line([center, vol_y_top, center, image_height - 1], fill=255)

        if save_dir:
            file_name = f"{ticker}_{start_date}_{end_date}_{period}D.png"
            save_path = os.path.join(save_dir, file_name)
            image.save(save_path, format='PNG')
            print(f"Chart saved to {save_path}")

        return image

    except Exception as e:
        print(f"Error generating chart for {ticker}: {e}")
        return None

def save_npz_file(images: List[np.ndarray], labels: List[int], meta_data: List[Dict], 
                  ticker: str, period: int, npz_file_name_prefix: str) -> None:
    images = np.array(images)
    labels = np.array(labels)
    npz_file_name = f"{npz_file_name_prefix}_{ticker}_{period}D.npz"
    np.savez_compressed(npz_file_name, images=images, labels=labels, meta_data=meta_data)
    print(f"Data saved to {npz_file_name}")

if __name__ == "__main__":
    stock_data_dir = "./data/stocks/"
    save_dir = "./data/charts"
    os.makedirs(save_dir, exist_ok=True)

    tickers = [f.split(".")[0] for f in os.listdir(stock_data_dir) if f.endswith(".csv")]

    # 모든 종목 데이터를 세로로 이어붙임
    dataframes = []
    for ticker in tickers:
        df = pd.read_csv(os.path.join(stock_data_dir, f"{ticker}.csv"), index_col="Date", parse_dates=True)
        df['Ticker'] = ticker  # Ticker 컬럼 추가
        dataframes.append(df)
    
    full_df = pd.concat(dataframes)

    # 로그 수익률 계산
    full_df['Log_Return'] = np.log(full_df['Adj Close'] / full_df['Adj Close'].shift(1))
    full_df.dropna(inplace=True)

    # 이미지 생성
    results_df = generate_candlestick_chart_images(
        full_df,
        tickers=tickers,
        periods=[20],
        has_ma=True,
        ma_period=5,
        has_volume=True,
        save_dir=save_dir,
        save_npz=True,
        npz_file_name_prefix="candlestick_data",
        only_sample=False,
        num_samples=5  # 각 기간별로 생성할 샘플 이미지 수
    )
    
    # 메타데이터 저장
    results_df.to_csv("data/metadata.csv", index=False)
    print("Metadata saved to data/metadata.csv")