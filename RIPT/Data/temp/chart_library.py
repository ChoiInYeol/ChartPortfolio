# chart_library.py
from __future__ import print_function, division
import numpy as np
import math
from PIL import Image
from skimage.draw import line
from Data import dgp_config as dcf


class DrawChartError(Exception):
    pass


class DrawOHLC(object):
    def __init__(self, df, has_volume_bar=False, ma_lags=None, chart_type="bar"):
        if np.around(df.iloc[0]["Close"], decimals=3) != 1.000:
            raise DrawChartError("Close on first day not equal to 1.")
        self.has_volume_bar = has_volume_bar
        self.vol = df["Vol"].values if has_volume_bar else None
        self.ma_lags = ma_lags
        self.ma_name_list = (
            ["ma" + str(ma_lag) for ma_lag in ma_lags] if ma_lags is not None else []
        )
        self.chart_type = chart_type
        assert chart_type in ["bar", "pixel", "centered_pixel"]

        if self.chart_type == "centered_pixel":
            self.df = self.centered_prices(df)
        else:
            self.df = df[["Open", "High", "Low", "Close"] + self.ma_name_list].abs()

        self.ohlc_len = len(df)
        assert self.ohlc_len in [5, 20, 60]
        self.minp = self.df.min().min()
        self.maxp = self.df.max().max()

        (
            self.ohlc_width,
            self.ohlc_height,
            self.volume_height,
        ) = self.__height_and_width()
        first_center = (dcf.BAR_WIDTH - 1) / 2.0
        self.centers = np.arange(
            first_center,
            first_center + dcf.BAR_WIDTH * self.ohlc_len,
            dcf.BAR_WIDTH,
            dtype=int,
        )

    def __height_and_width(self):
        width, height = dcf.IMAGE_WIDTH[self.ohlc_len], dcf.IMAGE_HEIGHT[self.ohlc_len]
        if self.has_volume_bar:
            volume_height = int(height / 5)
            height -= volume_height + dcf.VOLUME_CHART_GAP
        else:
            volume_height = 0
        return width, height, volume_height

    def __ret_to_yaxis(self, ret):
        pixels_per_unit = (self.ohlc_height - 1.0) / (self.maxp - self.minp)
        res = np.around((ret - self.minp) * pixels_per_unit)
        # Handle invalid values before casting to int
        res = np.where(np.isfinite(res), res, 0)
        return res.astype(int)

    def centered_prices(self, df):
        cols = ["Open", "High", "Low", "Close", "Prev_Close"] + self.ma_name_list
        df = df[cols].copy()
        df[cols] = df[cols].div(df["Close"], axis=0)
        df[cols] = df[cols].sub(df["Close"], axis=0)
        df.loc[df.index != 0, self.ma_name_list] = 0
        return df

    def draw_image(self, pattern_list=None):
        if self.maxp == self.minp or math.isnan(self.maxp) or math.isnan(self.minp):
            return None

        if self.chart_type == "centered_pixel":
            ohlc_image = self.__draw_centered_pixel_chart()
        else:
            ohlc_image = self.__draw_ohlc()

        if self.vol is not None:
            volume_image = self.__draw_vol()
            # 볼륨 이미지와 OHLC 이미지를 결합합니다.
            image_height = self.ohlc_height + self.volume_height + dcf.VOLUME_CHART_GAP
            image = np.full((image_height, self.ohlc_width), dcf.BACKGROUND_COLOR, dtype=np.uint8)
            image[self.volume_height + dcf.VOLUME_CHART_GAP:, :] = ohlc_image
            image[:self.volume_height, :] = volume_image
        else:
            image = ohlc_image

        if pattern_list is not None:
            # 필요한 경우 pattern_list 그리기 구현
            pass

        # 이미지를 수직으로 뒤집습니다.
        image = np.flipud(image)

        # Numpy 배열을 PIL 이미지로 변환합니다.
        image_pil = Image.fromarray(image, mode='L')

        return image_pil

    def __draw_vol(self):
        volume_image = np.full((self.volume_height, self.ohlc_width), dcf.BACKGROUND_COLOR, dtype=np.uint8)
        if self.vol is not None:
            vol_abs = np.abs(self.vol)
            valid_vol_mask = ~np.isnan(vol_abs)
            if np.any(valid_vol_mask):
                max_volume = np.nanmax(vol_abs)
                if not np.isnan(max_volume) and max_volume != 0:
                    pixels_per_volume = self.volume_height / max_volume
                    vol_heights = np.zeros_like(self.vol, dtype=int)
                    vol_heights[valid_vol_mask] = np.around(vol_abs[valid_vol_mask] * pixels_per_volume).astype(int)
                    x_indices = self.centers[valid_vol_mask].astype(int)
                    y_indices = vol_heights[valid_vol_mask] - 1
                    y_indices = np.clip(y_indices, 0, self.volume_height - 1)

                    if self.chart_type == "bar":
                        for x, y in zip(x_indices, y_indices):
                            volume_image[:y+1, x] = dcf.CHART_COLOR
                    elif self.chart_type in ["pixel", "centered_pixel"]:
                        volume_image[y_indices, x_indices] = dcf.CHART_COLOR
                    else:
                        raise ValueError(f"Chart type {self.chart_type} not supported")
        return volume_image


    def __draw_ohlc(self):
        ohlc_image = np.full((self.ohlc_height, self.ohlc_width), dcf.BACKGROUND_COLOR, dtype=np.uint8)

        # Y축 위치를 미리 계산합니다.
        open_y = self.__ret_to_yaxis(self.df["Open"].values).astype(int)
        high_y = self.__ret_to_yaxis(self.df["High"].values).astype(int)
        low_y = self.__ret_to_yaxis(self.df["Low"].values).astype(int)
        close_y = self.__ret_to_yaxis(self.df["Close"].values).astype(int)

        # 이동평균선 그리기
        for ma_name in self.ma_name_list:
            ma_values = self.df[ma_name].values
            ma_y = self.__ret_to_yaxis(ma_values)
            valid_indices = ~np.isnan(ma_y)
            
            x_indices = self.centers[valid_indices].astype(int)
            y_indices = ma_y[valid_indices].astype(int)
            if self.chart_type == "bar":
                # MA 포인트 간의 선을 그립니다.
                for j in range(len(x_indices) - 1):
                    x0, y0 = x_indices[j], y_indices[j]
                    x1, y1 = x_indices[j + 1], y_indices[j + 1]
                    # NaN 값 확인 및 건너뛰기
                    if np.isnan(y0) or np.isnan(y1):
                        continue

                    # 인덱스 클리핑
                    x0 = np.clip(x0, 0, self.ohlc_width - 1)
                    y0 = np.clip(y0, 0, self.ohlc_height - 1)
                    x1 = np.clip(x1, 0, self.ohlc_width - 1)
                    y1 = np.clip(y1, 0, self.ohlc_height - 1)

                    # 선 그리기
                    rr, cc = line(y0, x0, y1, x1)
                    valid_line_indices = (rr >= 0) & (rr < self.ohlc_height) & (cc >= 0) & (cc < self.ohlc_width)
                    ohlc_image[rr[valid_line_indices], cc[valid_line_indices]] = dcf.CHART_COLOR
            elif self.chart_type == "pixel":
                ohlc_image[y_indices, x_indices] = dcf.CHART_COLOR
            else:
                raise ValueError(f"Chart type {self.chart_type} not supported")

        # OHLC 막대 그리기 (기존 코드 유지)
        for i in range(self.ohlc_len):
            if np.isnan(high_y[i]) or np.isnan(low_y[i]):
                continue
            x = int(self.centers[i])
            left = int(x - dcf.BAR_WIDTH // 2)
            right = int(x + dcf.BAR_WIDTH // 2)

            if self.chart_type == "bar":
                # 인덱스 클리핑
                x = np.clip(x, 0, self.ohlc_width - 1)
                left = np.clip(left, 0, self.ohlc_width - 1)
                right = np.clip(right, 0, self.ohlc_width - 1)
                low_y[i] = np.clip(low_y[i], 0, self.ohlc_height - 1)
                high_y[i] = np.clip(high_y[i], 0, self.ohlc_height - 1)
                open_y[i] = np.clip(open_y[i], 0, self.ohlc_height - 1)
                close_y[i] = np.clip(close_y[i], 0, self.ohlc_height - 1)

                # 고가에서 저가까지의 선을 그립니다.
                ohlc_image[low_y[i]:high_y[i]+1, x] = dcf.CHART_COLOR
                # 시가와 종가의 수평선을 그립니다.
                if not np.isnan(open_y[i]):
                    ohlc_image[open_y[i], left:x+1] = dcf.CHART_COLOR
                if not np.isnan(close_y[i]):
                    ohlc_image[close_y[i], x:right+1] = dcf.CHART_COLOR
            elif self.chart_type == "pixel":
                # 픽셀 설정
                if 0 <= high_y[i] < self.ohlc_height and 0 <= x < self.ohlc_width:
                    ohlc_image[high_y[i], x] = dcf.CHART_COLOR
                if 0 <= low_y[i] < self.ohlc_height and 0 <= x < self.ohlc_width:
                    ohlc_image[low_y[i], x] = dcf.CHART_COLOR
                # 시가와 종가의 수평선
                if not np.isnan(open_y[i]):
                    ohlc_image[open_y[i], left:x+1] = dcf.CHART_COLOR
                if not np.isnan(close_y[i]):
                    ohlc_image[close_y[i], x:right+1] = dcf.CHART_COLOR
            else:
                raise ValueError(f"Chart type {self.chart_type} not supported")

        return ohlc_image

    def __draw_centered_pixel_chart(self):
        ohlc_image = np.full((self.ohlc_height, self.ohlc_width), dcf.BACKGROUND_COLOR, dtype=np.uint8)

        # Y축 위치를 미리 계산합니다.
        high_y = self.__ret_to_yaxis(self.df["High"].values)
        low_y = self.__ret_to_yaxis(self.df["Low"].values)
        prev_close_y = self.__ret_to_yaxis(self.df["Prev_Close"].values)
        open_y = self.__ret_to_yaxis(self.df["Open"].values)

        # 픽셀 그리기
        for i in range(self.ohlc_len):
            x = int(self.centers[i])
            if not np.isnan(high_y[i]):
                ohlc_image[high_y[i], x] = dcf.CHART_COLOR
            if not np.isnan(low_y[i]):
                ohlc_image[low_y[i], x] = dcf.CHART_COLOR

            left = int(x - dcf.BAR_WIDTH // 2)
            right = int(x + dcf.BAR_WIDTH // 2)

            if not np.isnan(open_y[i]):
                ohlc_image[open_y[i], left:x+1] = dcf.CHART_COLOR
            if not np.isnan(prev_close_y[i]):
                ohlc_image[prev_close_y[i], left:right+1] = dcf.CHART_COLOR

        # 이동평균선이 있는 경우 그리기
        for ma_name in self.ma_name_list:
            ma_values = self.df[ma_name].values
            ma_y = self.__ret_to_yaxis(ma_values)
            valid_indices = ~np.isnan(ma_y)
            x_indices = self.centers[valid_indices].astype(int)
            y_indices = ma_y[valid_indices]
            ohlc_image[y_indices, x_indices] = dcf.CHART_COLOR

        return ohlc_image
