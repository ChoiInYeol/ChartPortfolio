# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import os.path as op
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from numba import jit, prange, njit
import cProfile
import pstats
import io
import random

from Data import equity_data as eqd
from Data import chart_library as cl
from Data import dgp_config as dcf
from Misc import utilities as ut

from typing import List, Optional, Dict


class ChartGenerationError(Exception):
    pass


@jit(nopython=True)
def adjust_price_numba(close, open_, high, low, ret):
    n = len(close)
    res_close = np.empty(n)
    res_open = np.empty(n)
    res_high = np.empty(n)
    res_low = np.empty(n)

    res_close[0] = 1.0
    res_open[0] = abs(open_[0]) / abs(close[0])
    res_high[0] = abs(high[0]) / abs(close[0])
    res_low[0] = abs(low[0]) / abs(close[0])

    pre_close = 1.0
    for i in range(1, n):
        res_close[i] = (1 + ret[i]) * pre_close
        res_open[i] = res_close[i] / abs(close[i]) * abs(open_[i])
        res_high[i] = res_close[i] / abs(close[i]) * abs(high[i])
        res_low[i] = res_close[i] / abs(close[i]) * abs(low[i])
        
        if not np.isnan(res_close[i]):
            pre_close = res_close[i]

    return res_close, res_open, res_high, res_low


def adjust_price(df):
    if len(df) == 0:
        raise ChartGenerationError("adjust_price: Empty Dataframe")
    if len(df.Date.unique()) != len(df):
        raise ChartGenerationError("adjust_price: Dates not unique")

    if df.iloc[0]["Close"] == 0.0 or pd.isna(df.iloc[0]["Close"]):
        raise ChartGenerationError("adjust_price: First day close is nan or zero")

    # Convert necessary columns to Numpy arrays
    close = df["Close"].values
    open_ = df["Open"].values
    high = df["High"].values
    low = df["Low"].values
    ret = df["Ret"].values

    # Call the Numba-optimized function
    close_adj, open_adj, high_adj, low_adj = adjust_price_numba(close, open_, high, low, ret)

    df = df.copy()  # 슬라이스 경고를 피하기 위해 복사본 생성
    df.loc[:, "Close"] = close_adj
    df.loc[:, "Open"] = open_adj
    df.loc[:, "High"] = high_adj
    df.loc[:, "Low"] = low_adj

    return df


@njit
def convert_daily_df_to_chart_freq_df_numba(daily_data, dates, chart_freq):
    n_cols = daily_data.shape[1]
    ohlc_len = len(daily_data) // chart_freq
    result = np.empty((ohlc_len, n_cols))
    result_dates = np.empty(ohlc_len, dtype=dates.dtype)

    for i in range(ohlc_len):
        start = i * chart_freq
        end = (i + 1) * chart_freq
        subdata = daily_data[start:end]

        result[i] = subdata[-1]  # Copy last row
        result[i, 1] = subdata[0, 1]  # Open
        result[i, 2] = np.max(subdata[:, 2])  # High
        result[i, 3] = np.min(subdata[:, 3])  # Low
        result[i, 5] = np.sum(subdata[:, 5])  # Vol
        result[i, 6] = np.prod(1 + subdata[:, 6]) - 1  # Ret

        result_dates[i] = dates[end - 1]  # Use the last date in the period

    return result, result_dates


def convert_daily_df_to_chart_freq_df(daily_df, chart_freq):
    if not len(daily_df) % chart_freq == 0:
        raise ChartGenerationError("df not divided by chart freq")

    values = daily_df.values
    dates = daily_df['Date'].values

    result_array, result_dates = convert_daily_df_to_chart_freq_df_numba(values, dates, chart_freq)

    df = pd.DataFrame(result_array, columns=daily_df.columns)
    df['Date'] = result_dates

    return df


def load_adjusted_daily_prices(stock_df, date, window_size, ma_lags, chart_freq, chart_len, need_adjust_price):
    if date not in set(stock_df.Date):
        return 0
    date_index = stock_df[stock_df.Date == date].index[0]
    ma_offset = 0 if ma_lags is None else np.max(ma_lags)
    data = stock_df.loc[
        (date_index - (window_size - 1) - ma_offset) : date_index
    ]
    if len(data) < window_size:
        return 1
    if len(data) < (window_size + ma_offset):
        ma_lags = []
        data = stock_df.loc[(date_index - (window_size - 1)) : date_index]
    else:
        ma_lags = ma_lags
    if chart_freq != 1:
        data = convert_daily_df_to_chart_freq_df(data, chart_freq)
    if need_adjust_price:
        if data["Close"].iloc[0] == 0.0 or pd.isna(data["Close"].iloc[0]):
            return 2
        data = adjust_price(data)
    else:
        data = data.copy()
    start_date_index = data.index[-1] - chart_len + 1
    if data["Close"].loc[start_date_index] == 0 or np.isnan(
        data["Close"].loc[start_date_index]
    ):
        return 3
    data[["Open", "High", "Low", "Close"]] *= (
        1.0 / data["Close"].loc[start_date_index]
    )
    if ma_lags is not None:
        ma_name_list = ["ma" + str(lag) for lag in ma_lags]
        for i, ma_name in enumerate(ma_name_list):
            chart_num = int(ma_lags[i] / chart_freq)
            data[ma_name] = (
                data["Close"].rolling(chart_num, min_periods=1).mean()
            )
    data["Prev_Close"] = data["Close"].shift(1)
    df = data.loc[start_date_index:]
    if (
        len(df) != chart_len
        or np.around(df.iloc[0]["Close"], decimals=3) != 1.000
    ):
        return 4
    df = df.reset_index(drop=True)
    return df



def generate_daily_features(stock_df, date, window_size, ma_lags, chart_freq, chart_len, volume_bar, chart_type, need_adjust_price, ret_len_list):
    res = load_adjusted_daily_prices(stock_df, date, window_size, ma_lags, chart_freq, chart_len, need_adjust_price)
    if isinstance(res, int):
        return res
    df = res
    
    # 데이터 유효성 검사 추가
    df = df.bfill()
    
    ma_lags_present = [int(ma_col[2:]) for ma_col in df.columns if "ma" in ma_col]
    ohlc_obj = cl.DrawOHLC(
        df,
        has_volume_bar=volume_bar,
        ma_lags=ma_lags_present,
        chart_type=chart_type,
    )
    image_data = ohlc_obj.draw_image()
    if image_data is None:
        return 5

    last_day = df[df.Date == date].iloc[0]
    feature_dict = {feature: last_day[feature] for feature in stock_df.columns if feature in last_day.index}
    ret_list = ["Ret"] + [f"Ret_{i}d" for i in ret_len_list]
    for ret in ret_list:
        if ret in feature_dict:
            feature_dict[f"{ret}_label"] = 1 if feature_dict[ret] > 0 else 0 if feature_dict[ret] <= 0 else 2
            vol = feature_dict.get("EWMA_vol", 0)
            feature_dict[f"{ret}_tstat"] = 0 if (vol == 0 or pd.isna(vol)) else feature_dict[ret] / vol

    feature_dict['image'] = image_data
    feature_dict['window_size'] = window_size
    feature_dict['Date'] = date

    return feature_dict


def process_stock(stock_id, df, freq, year, window_size, ma_lags, chart_freq, chart_len, volume_bar, chart_type, need_adjust_price, ret_len_list):
    """주식별 차트 이미지와 레이블을 생성하는 함수
    
    Args:
        stock_id (int): 주식 종목 코드
        df (pd.DataFrame): 주가 데이터
        freq (str): 데이터 샘플링 주기. 'day'인 경우 매일 슬라이딩 윈도우로 샘플 생성
        year (int): 데이터를 생성할 연도
        window_size (int): 차트 생성에 사용할 기간 (일 단위)
        ma_lags (List[int] | None): 이동평균선 기간 리스트 (예: [5, 20, 60])
        chart_freq (int): 차트의 봉 주기 (1: 일봉, 5: 주봉 등)
        chart_len (int): 차트 길이 (일 단위)
        volume_bar (bool): 거래량 바 표시 여부
        chart_type (str): 차트 유형 ("bar", "pixel", "centered_pixel" 중 하나)
        need_adjust_price (bool): 수정주가 적용 여부
        ret_len_list (List[int]): 수익률 계산 기간 리스트
    
    Returns:
        List[dict]: 주식별 차트 이미지와 레이블 데이터
    """
    stock_df = df.xs(stock_id, level=1).copy()
    stock_df = stock_df.reset_index()
    
    # freq가 'day'인 경우 window_size 이상의 데이터가 있는 모든 날짜를 대상으로 함
    if freq == "day":
        # 해당 연도의 모든 거래일을 대상으로
        year_dates = stock_df[stock_df.Date.dt.year == year].Date
        # window_size만큼의 과거 데이터가 있는 날짜만 선택
        valid_dates = []
        for date in year_dates:
            end_idx = stock_df[stock_df.Date <= date].index[-1]
            if end_idx >= window_size - 1:  # 충분한 과거 데이터가 있는지 확인
                valid_dates.append(date)
    else:
        # 기존 로직: period_end_dates 사용
        dates = eqd.get_period_end_dates(freq)
        valid_dates = dates[dates.year == year]
    
    results = []
    for date in valid_dates:
        try:
            image_label_data = generate_daily_features(
                stock_df, date, window_size, ma_lags, chart_freq, chart_len, 
                volume_bar, chart_type, need_adjust_price, ret_len_list
            )
            if isinstance(image_label_data, dict):
                image_label_data["StockID"] = stock_id
                results.append(image_label_data)
            elif isinstance(image_label_data, int):
                results.append(("miss", image_label_data))
        except ChartGenerationError:
            print(f"DGP Error on {stock_id} {date}")
            
    return results


@njit
def calculate_ma(close, ma_lags):
    n = len(close)
    num_lags = len(ma_lags)
    result = np.empty((num_lags, n), dtype=np.float32)
    cumsum = np.zeros(n + 1)  # 누적합 배열 생성

    # cumsum 계산: cumsum[i+1] = cumsum[i] + close[i]
    for i in range(n):
        cumsum[i + 1] = cumsum[i] + close[i]

    for i in range(num_lags):
        lag = ma_lags[i]
        if lag <= n:
            result[i, :lag-1] = np.nan
            result[i, lag-1:] = (cumsum[lag:] - cumsum[:-lag]) / lag
        else:
            result[i, :] = np.nan

    return result



def generate_daily_ts_features(stock_df, date, window_size, ma_lags, ret_len_list):
    if date not in set(stock_df.Date):
        return 0

    date_index = stock_df[stock_df.Date == date].index[0]

    ma_offset = 0 if ma_lags is None else np.max(ma_lags) - 1
    data = stock_df.loc[
        (date_index - (window_size - 1) - ma_offset) : date_index
    ]

    if len(data) != (window_size + ma_offset):
        return 1

    if data["Close"].iloc[0] == 0.0 or pd.isna(data["Close"].iloc[0]):
        return 2
    data = adjust_price(data)

    start_date_index = data.index[-1] - window_size + 1
    if data["Close"].loc[start_date_index] == 0 or np.isnan(
        data["Close"].loc[start_date_index]
    ):
        return 3

    data[["Open", "High", "Low", "Close"]] *= (
        1.0 / data["Close"].loc[start_date_index]
    )

    if ma_lags is not None:
        ma_values = calculate_ma(data["Close"].values, ma_lags)
        for i, ma_name in enumerate(["ma" + str(lag) for lag in ma_lags]):
            data[ma_name] = ma_values[i]

    window = data.loc[start_date_index:]

    if (
        len(window) != window_size
        or np.around(window.iloc[0]["Close"], decimals=3) != 1.000
    ):
        return 4

    window = window.reset_index(drop=True)

    predictors = window[
        ["Open", "High", "Low", "Close", "Vol"]
    ].T.to_numpy()

    last_day = data[data.Date == date]
    assert len(last_day) == 1
    last_day = last_day.iloc[0]

    feature_list = [
        "StockID",
        "Date",
        "EWMA_vol",
        "Ret",
        "MarketCap",
    ] + [f"Ret_{i}d" for i in ret_len_list]
    feature_dict = {feature: last_day[feature] for feature in feature_list if feature in last_day}

    ret_list = ["Ret"] + [f"Ret_{i}d" for i in ret_len_list]
    for ret in ret_list:
        if ret in feature_dict:
            feature_dict["{}_label".format(ret)] = (
                1 if feature_dict[ret] > 0 else 0 if feature_dict[ret] <= 0 else 2
            )
            vol = feature_dict.get("EWMA_vol", 0)
            feature_dict["{}_tstat".format(ret)] = (
                0 if (vol == 0 or pd.isna(vol)) else feature_dict[ret] / vol
            )
    feature_dict["predictor"] = predictors
    feature_dict["window_size"] = window_size

    return feature_dict


def process_stock_ts(stock_id, df, freq, year, window_size, ma_lags, ret_len_list):
    stock_df = df.xs(stock_id, level=1).copy()
    stock_df = stock_df.reset_index()
    stock_df['StockID'] = stock_id
    
    # freq가 'day'인 경우 모든 거래일을 dates로 사용
    if freq == "day":
        dates = stock_df[stock_df.Date.dt.year == year].Date
    else:
        dates = stock_df[~pd.isna(stock_df[f"Ret_{freq}"])].Date
        dates = dates[dates.dt.year == year]
    
    results = []
    for date in dates:
        try:
            predictor_label_data = generate_daily_ts_features(stock_df, date, window_size, ma_lags, ret_len_list)
            if isinstance(predictor_label_data, dict):
                results.append(predictor_label_data)
            elif isinstance(predictor_label_data, int):
                results.append(("miss", predictor_label_data))
        except ChartGenerationError:
            continue
    return results


class GenerateStockData(object):
    """주식 차트 이미지와 시계열 데이터를 생성하고 저장하는 클래스
    
    Attributes:
        country (str): 대상 국가 코드 (예: "USA")
        year (int): 데이터를 생성할 연도
        window_size (int): 차트 생성에 사용할 기간 (일 단위)
        freq (str): 데이터 샘플링 주기 ("day", "week", "month", "quarter")
        chart_freq (int): 차트의 봉 주기 (1: 일봉, 5: 주봉 등)
        ma_lags (List[int] | None): 이동평균선 기간 리스트 (예: [5, 20, 60])
        volume_bar (bool): 거래량 바 표시 여부
        need_adjust_price (bool): 수정주가 적용 여부
        allow_tqdm (bool): 진행률 표시 여부
        chart_type (str): 차트 유형 ("bar", "pixel", "centered_pixel" 중 하나)
        ret_len_list (List[int]): 수익률 계산 기간 리스트
        bar_width (int): 차트 봉의 너비 (픽셀)
        image_width (Dict[int, int]): 차트 길이별 이미지 너비 매핑
        image_height (Dict[int, int]): 차트 길이별 이미지 높이 매핑
        width (int): 현재 설정된 이미지 너비
        height (int): 현재 설정된 이미지 높이
        df (pd.DataFrame | None): 로드된 주가 데이터
        stock_id_list (np.ndarray | None): 처리할 종목 코드 리스트
        save_dir (str): 이미지 데이터 저장 경로
        save_dir_ts (str): 시계열 데이터 저장 경로
        image_save_dir (str): 샘플 이미지 저장 경로
        file_name (str): 생성될 파일의 기본 이름
        log_file_name (str): 로그 파일 경로
        labels_filename (str): 레이블 데이터 파일 경로
        images_filename (str): 이미지 데이터 파일 경로
    
    Class Attributes:
        _data_cache (Dict[int, pd.DataFrame]): 연도별 데이터 캐시
    """
    
    _data_cache: Dict[int, pd.DataFrame] = {}
     
    def __init__(
        self,
        country: str,
        year: int,
        window_size: int,
        freq: str,
        chart_freq: int = 1,
        ma_lags: Optional[List[int]] = None,
        volume_bar: bool = False,
        need_adjust_price: bool = True,
        allow_tqdm: bool = False,
        chart_type: str = "bar",
    ) -> None:
        """
        Args:
            country: 대상 국가 코드
            year: 데이터를 생성할 연도
            window_size: 차트 생성에 사용할 기간 (일 단위)
            freq: 데이터 샘플링 주기 ("day", "week", "month", "quarter")
            chart_freq: 차트의 봉 주기 (1: 일봉, 5: 주봉 등)
            ma_lags: 이동평균선 기간 리스트
            volume_bar: 거래량 바 표시 여부
            need_adjust_price: 수정주가 적용 여부
            allow_tqdm: 진행률 표시 여부
            chart_type: 차트 유형 ("bar", "pixel", "centered_pixel")
            
        Raises:
            AssertionError: freq가 유효하지 않은 값일 경우
            AssertionError: window_size가 chart_freq로 나누어떨어지지 않을 경우
            AssertionError: chart_len이 [5, 20, 60] 중 하나가 아닐 경우
            AssertionError: chart_type이 유효하지 않은 값일 경우
        """
        self.country = country
        self.year = year
        self.window_size = window_size
        self.freq = freq
        assert self.freq in ["day", "week", "month", "quarter"], f"Invalid freq: {self.freq}. Must be one of ['day', 'week', 'month', 'quarter']"
        self.chart_freq = chart_freq
        assert window_size % chart_freq == 0
        self.chart_len = int(window_size / chart_freq)
        assert self.chart_len in [5, 20, 60]
        self.ma_lags = ma_lags
        self.volume_bar = volume_bar
        self.need_adjust_price = need_adjust_price
        self.allow_tqdm = allow_tqdm
        assert chart_type in ["bar", "pixel", "centered_pixel"]
        self.chart_type = chart_type

        self.ret_len_list = [5, 20, 60, 65, 180, 250, 260]
        self.bar_width = 3
        self.image_width = {
            5: self.bar_width * 5,
            20: self.bar_width * 20,
            60: self.bar_width * 60,
        }
        self.image_height = {5: 32, 20: 64, 60: 96}

        self.width, self.height = (
            self.image_width[int(self.chart_len)],
            self.image_height[int(self.chart_len)],
        )

        self.df = None
        self.stock_id_list = None

        self.save_dir = ut.get_dir(
            op.join(dcf.STOCKS_SAVEPATH, f"stocks_{self.country}/dataset_all")
        )
        self.save_dir_ts = ut.get_dir(
            op.join(dcf.STOCKS_SAVEPATH, f"stocks_{self.country}_ts/dataset_all")
        )
        
        self.image_save_dir = ut.get_dir(op.join(dcf.STOCKS_SAVEPATH, "sample_images"))
        vb_str = "has_vb" if self.volume_bar else "no_vb"
        ohlc_len_str = "" if self.chart_freq == 1 else f"_{self.chart_len}ohlc"
        chart_type_str = "" if self.chart_type == "bar" else f"{self.chart_type}_"
        self.file_name = f"{chart_type_str}{self.window_size}d_{self.freq}_{vb_str}_{str(self.ma_lags)}_ma_{self.year}{ohlc_len_str}"
        self.log_file_name = op.join(self.save_dir, f"{self.file_name}.txt")
        self.labels_filename = op.join(
            self.save_dir, f"{self.file_name}_labels.feather"
        )
        self.images_filename = op.join(self.save_dir, f"{self.file_name}_data.npz")

    def _get_feature_and_dtype_list(self):
        float32_features = [
            "EWMA_vol",
            "Ret",
            "MarketCap",
        ] + [f"Ret_{i}d" for i in self.ret_len_list]
        int8_features = ["window_size"]
        object_features = ["StockID"]
        datetime_features = ["Date"]
        uint8_features = ["image"]  # 이미지 데이터 타입 추가
        feature_list = (
            float32_features + int8_features + object_features + datetime_features + uint8_features
        )
        float32_dict = {feature: np.float32 for feature in float32_features}
        int8_dict = {feature: np.int8 for feature in int8_features}
        object_dict = {feature: object for feature in object_features}
        datetime_dict = {feature: "datetime64[ns]" for feature in datetime_features}
        uint8_dict = {feature: np.uint8 for feature in uint8_features}  # 이미지 데이터 타입 추가
        dtype_dict = {**float32_dict, **int8_dict, **object_dict, **datetime_dict, **uint8_dict}
        return dtype_dict, feature_list

    def save_annual_data(self):
        if (
            op.isfile(self.log_file_name)
            and op.isfile(self.images_filename)
        ):
            print("Found pregenerated file {}".format(self.file_name))
            return
        print(f"Generating {self.file_name}")

        # 기존 파일 삭제
        if os.path.exists(self.images_filename):
            os.remove(self.images_filename)
        if os.path.exists(self.log_file_name):
            os.remove(self.log_file_name)

        # 데이터 로드 시 캐시 확인
        if self.year in GenerateStockData._data_cache:
            self.df = GenerateStockData._data_cache[self.year]
            print(f"Using cached data for year {self.year}")
        else:
            self.df = eqd.get_processed_US_data_by_year(self.year)
            GenerateStockData._data_cache[self.year] = self.df
            print(f"Loaded data for year {self.year} and cached it")

        self.stock_id_list = np.unique(self.df.index.get_level_values("StockID"))
        dtype_dict, feature_list = self._get_feature_and_dtype_list()
        data_miss = np.zeros(7)
        sample_num = 0
        samples = []

        # 멀티프로세싱 실행
        process_stock_partial = partial(
            process_stock,
            df=self.df,
            freq=self.freq,
            year=self.year,
            window_size=self.window_size,
            ma_lags=self.ma_lags,
            chart_freq=self.chart_freq,
            chart_len=self.chart_len,
            volume_bar=self.volume_bar,
            chart_type=self.chart_type,
            need_adjust_price=self.need_adjust_price,
            ret_len_list=self.ret_len_list
        )
        
        chunksize = max(1, len(self.stock_id_list) // (mp.cpu_count() * 4))
        with mp.Pool(processes=mp.cpu_count()) as pool:
            all_results = list(tqdm(
                pool.imap(process_stock_partial, self.stock_id_list, chunksize=chunksize),
                total=len(self.stock_id_list), disable=not self.allow_tqdm))

        # 결과 처리
        for stock_results in all_results:
            for result in stock_results:
                if isinstance(result, dict):
                    samples.append(result)
                    sample_num += 1
                elif isinstance(result, tuple) and result[0] == "miss":
                    data_miss[result[1]] += 1

        if sample_num == 0:
            print("No valid samples generated. Skipping saving.")
            return

        # 데이터프레임 생성
        df = pd.DataFrame(samples)
        # 이미지 데이터를 별도 컬럼으로 추출
        images = df.pop('image').tolist()
        # 이미지를 numpy 배열로 변환
        images_array = np.array([np.array(img).flatten() for img in images], dtype=np.uint8)

        # 이미지와 라벨을 함께 저장
        np.savez_compressed(self.images_filename, images=images_array, labels=df.to_dict('list'))

        print(f"Save data to {self.images_filename}")
        
        # 로그 파일 작성 수정
        log_file = open(self.log_file_name, "w+")
        log_file.write(
            "total_dates:%d total_missing:%d type0:%d type1:%d type2:%d type3:%d type4:%d type5:%d type6:%d"
            % (
                sample_num,
                int(np.sum(data_miss)),
                int(data_miss[0]),
                int(data_miss[1]),
                int(data_miss[2]),
                int(data_miss[3]),
                int(data_miss[4]),
                int(data_miss[5]),
                int(data_miss[6]),
            )
        )
        log_file.close()
        print(f"Save log file to {self.log_file_name}")

        # 랜덤하게 샘플 이미지 선택 및 저장
        if images:
            random_index = random.randint(0, len(images) - 1)
            stock_id = df.iloc[random_index]['StockID']
            date = df.iloc[random_index]['Date']
            image = images[random_index]
            image_pil = image  # 이미지는 PIL 이미지 객체로 저장되어 있음
            image_pil.save(
                op.join(
                    self.image_save_dir,
                    f"{self.file_name}_{stock_id}_{date.strftime('%Y%m%d')}.png",
                )
            )

    def get_ts_feature_and_dtype_list(self):
        float32_features = [
            "EWMA_vol",
            "Ret",
            "MarketCap",
        ] + [f"Ret_{i}d" for i in self.ret_len_list]
        int8_features = [
            "Ret_label",
            "window_size",
        ] + [f"Ret_{i}d_label" for i in self.ret_len_list]
        object_features = ["StockID"]
        datetime_features = ["Date"]
        float32_features.append("predictor")
        feature_list = (
            float32_features + int8_features + object_features + datetime_features
        )
        float32_dict = {feature: np.float32 for feature in float32_features}
        int8_dict = {feature: np.int8 for feature in int8_features}
        object_dict = {feature: object for feature in object_features}
        datetime_dict = {feature: "datetime64[ns]" for feature in datetime_features}
        dtype_dict = {**float32_dict, **int8_dict, **object_dict, **datetime_dict}
        return dtype_dict, feature_list

    def save_annual_ts_data(self):
        file_name = "{}d_{}_{}_vb_{}_ma_{}_ts".format(
            self.window_size,
            self.freq,
            "has" if self.volume_bar else "no",
            str(self.ma_lags),
            self.year,
        )
        
        log_file_name = os.path.join(self.save_dir_ts, "{}.txt".format(file_name))
        data_filename = os.path.join(self.save_dir_ts, "{}_data_new.npz".format(file_name))
        if os.path.isfile(log_file_name) and os.path.isfile(data_filename):
            print("Found pregenerated file {}".format(file_name))
            return
        
        print(f"Generating {file_name}")

        # 기존 파일 삭제
        if os.path.exists(data_filename):
            os.remove(data_filename)
        if os.path.exists(log_file_name):
            os.remove(log_file_name)

        # 데이터 로드 시 캐시 확인
        if self.year in GenerateStockData._data_cache:
            self.df = GenerateStockData._data_cache[self.year]
            print(f"Using cached data for year {self.year}")
        else:
            self.df = eqd.get_processed_US_data_by_year(self.year)
            GenerateStockData._data_cache[self.year] = self.df
            print(f"Loaded data for year {self.year} and cached it")

        self.stock_id_list = np.unique(self.df.index.get_level_values("StockID"))
        
        dtype_dict, feature_list = self.get_ts_feature_and_dtype_list()
        data_miss = np.zeros(7)
        sample_num = 0
        samples = []

        # 멀티프로세싱 실행
        process_stock_ts_partial = partial(
            process_stock_ts,
            df=self.df,
            freq=self.freq,
            year=self.year,
            window_size=self.window_size,
            ma_lags=self.ma_lags,
            ret_len_list=self.ret_len_list
        )
        chunksize = max(1, len(self.stock_id_list) // (mp.cpu_count() * 4))
        with mp.Pool(processes=mp.cpu_count()) as pool:
            all_results = list(tqdm(pool.imap(process_stock_ts_partial, self.stock_id_list, chunksize=chunksize), total=len(self.stock_id_list), disable=not self.allow_tqdm))

        # 결과 처리
        for stock_results in all_results:
            for result in stock_results:
                if isinstance(result, dict):
                    samples.append(result)
                    sample_num += 1
                elif isinstance(result, tuple) and result[0] == "miss":
                    data_miss[result[1]] += 1

        if sample_num == 0:
            print("No valid samples generated. Skipping saving.")
            return

        # 데이터프레임 생성
        df = pd.DataFrame(samples)
        predictors = df.pop('predictor').tolist()
        predictors_array = np.array(predictors, dtype=np.float32)

        # 데이터 저장
        np.savez_compressed(
            data_filename, predictors=predictors_array, labels=df.to_dict('list')
        )
        print(f"Save data to {data_filename}")

        log_file = open(log_file_name, "w+")
        log_file.write(
            "total_dates:%d total_missing:%d type0:%d type1:%d type2:%d type3:%d type4:%d type5:%d"
            % (
                sample_num,
                sum(data_miss),
                data_miss[0],
                data_miss[1],
                data_miss[2],
                data_miss[3],
                data_miss[4],
                data_miss[5],
            )
        )
        log_file.close()
        print(f"Save log file to {log_file_name}")


def profile_function(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())
        return result
    return wrapper

@profile_function
def main():
    generator = GenerateStockData("USA", 2020, 60, "month")
    generator.save_annual_data()
    generator.save_annual_ts_data()

if __name__ == "__main__":
    main()
