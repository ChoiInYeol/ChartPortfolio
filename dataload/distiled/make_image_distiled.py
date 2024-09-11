import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from typing import List, Tuple

def generate_candlestick_chart_images(
    df: pd.DataFrame,
    tickers: List[str],
    periods: List[int],
    save_dir: str = "./data/charts",
    has_ma: bool = False,
    has_volume: bool = False,
    ma_period: int = 5,
    save_npz: bool = False,
    npz_file_name_prefix: str = "candlestick_data"
) -> pd.DataFrame:
    results = []
    size_dict = {20: (64, 60), 60: (96, 180), 180: (128, 360), 240: (160, 480)}

    for ticker in tickers:
        ticker_df = df[df['Ticker'] == ticker]
        ticker_df = ticker_df.sort_index()  # 날짜순으로 정렬

        for period in periods:
            images = []
            labels = []
            meta_data = []

            start_idx = 0
            while start_idx < len(ticker_df) - period + 1:
                ma_start_idx = max(0, start_idx - ma_period + 1)
                sub_df = ticker_df.iloc[ma_start_idx:start_idx + period]

                start_date = sub_df.index[ma_period - 1].strftime('%Y-%m-%d')
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
                    save_dir=save_dir if not save_npz else None
                )

                if image is not None:
                    log_return = sub_df['Log_Return'].iloc[-1]
                    label = 1 if log_return > 0 else 0

                    if save_npz:
                        images.append(np.array(image))
                        labels.append(label)
                        meta_data.append({
                            'Ticker': ticker,
                            'Start_Date': start_date,
                            'End_Date': end_date,
                            'Period': period
                        })
                    else:
                        results.append({
                            'Ticker': ticker,
                            'Start_Date': start_date,
                            'End_Date': end_date,
                            'Period': period,
                            'Image_Path': f"{save_dir}/{ticker}_{start_date}_{end_date}_{period}D.png",
                            'Label': label
                        })

                start_idx += period  # Period 크기만큼 이동

            # 각 period별로 .npz 파일 저장
            if save_npz and images:
                images = np.array(images)
                labels = np.array(labels)
                npz_file_name = f"{save_dir}/{npz_file_name_prefix}_{ticker}_{period}D.npz"
                np.savez_compressed(npz_file_name, images=images, labels=labels, meta_data=meta_data)
                print(f"Data saved to {npz_file_name}")

    return pd.DataFrame(results)

def generate_single_candlestick_chart(
    df: pd.DataFrame,
    ticker: str,
    start_date: str,
    end_date: str,
    period: int,
    has_ma: bool = False,
    has_volume: bool = False,
    ma_period: int = 5,
    size: Tuple[int, int] = (64, 60),
    save_dir: str = "./data/charts"
) -> Image.Image:
    try:
        ohlc_len = period
        ohlc_height, base_image_width = size
        volume_height = 32 if has_volume else 0
        bar_width = 3
        gap_width = 3  # 캔들스틱 간 간격

        total_width_per_candle = bar_width + gap_width
        image_width = ohlc_len * total_width_per_candle - gap_width

        image_height = ohlc_height + volume_height
        image = Image.new("L", (image_width, image_height), color=0)
        draw = ImageDraw.Draw(image)

        centers = np.arange(bar_width // 2, image_width, total_width_per_candle, dtype=int)

        min_price = df[['Open', 'High', 'Low', 'Close']].iloc[-period:].min().min()
        max_price = df[['Open', 'High', 'Low', 'Close']].iloc[-period:].max().max()

        def price_to_y(price):
            return int(ohlc_height * (1 - (price - min_price) / (max_price - min_price)))

        for i in range(-period, 0):
            open_y = price_to_y(df.iloc[i]['Open'])
            high_y = price_to_y(df.iloc[i]['High'])
            low_y = price_to_y(df.iloc[i]['Low'])
            close_y = price_to_y(df.iloc[i]['Close'])

            draw.line([(centers[i + period], low_y), (centers[i + period], high_y)], fill=255)

            if open_y < close_y:
                draw.rectangle([centers[i + period] - 1, open_y, centers[i + period] + 1, close_y], fill=255)
            else:
                draw.rectangle([centers[i + period] - 1, close_y, centers[i + period] + 1, open_y], fill=255)

        if has_ma:
            ma_values = df['Close'].rolling(window=ma_period).mean()
            ma_values = ma_values.dropna().iloc[-period:]
            ma_x = centers[-len(ma_values):]

            for i in range(1, len(ma_values)):
                draw.line([(ma_x[i-1], price_to_y(ma_values.iloc[i-1])), (ma_x[i], price_to_y(ma_values.iloc[i]))], fill=255, width=1)

        if has_volume:
            max_volume = df['Volume'].iloc[-period:].max()
            volume_y_start = ohlc_height

            for i in range(-period, 0):
                vol_bar_height = int(volume_height * (df.iloc[i]['Volume'] / max_volume))
                vol_y_top = volume_y_start + volume_height - vol_bar_height
                draw.rectangle([centers[i + period] - 1, vol_y_top, centers[i + period] + 1, volume_y_start + volume_height], fill=255)

        if save_dir:
            file_name = f"{ticker}_{start_date}_{end_date}_{period}D.png"
            save_path = os.path.join(save_dir, file_name)
            image.save(save_path, format='PNG')
            print(f"Chart saved to {save_path}")

        return image

    except Exception as e:
        print(f"Error generating chart for {ticker}: {e}")
        return None

if __name__ == "__main__":
    stock_data_dir = "data/stocks/"
    save_dir = "data/charts"
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
    generate_candlestick_chart_images(
        full_df,
        tickers=tickers,
        periods=[20],
        save_dir=save_dir,
        save_npz=True,
        npz_file_name_prefix="candlestick_data"
    )
