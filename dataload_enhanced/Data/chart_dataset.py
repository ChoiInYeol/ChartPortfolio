# -*- coding: utf-8 -*-
import pandas as pd
import os.path as op
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from Data import dgp_config as dcf
from Data import equity_data as eqd
from Misc import utilities as ut


class EquityDataset(Dataset):
    """
    주식 데이터셋을 위한 클래스입니다.
    
    이 클래스는 주식 차트 이미지와 관련 레이블을 로드하고 처리합니다.
    PyTorch의 Dataset 클래스를 상속받아 구현되었습니다.

    Attributes:
        ws (int): 윈도우 크기
        pw (int): 예측 윈도우 크기
        freq (str): 데이터 빈도 ('week', 'month', 'quarter', 'year' 중 하나)
        year (int): 데이터 연도
        ohlc_len (int): OHLC(Open, High, Low, Close) 데이터 길이
        data_freq (str): 실제 데이터 빈도
        country (str): 국가 코드
        has_vb (bool): 거래량 바 포함 여부
        has_ma (bool): 이동평균선 포함 여부
        chart_type (str): 차트 유형 ('bar', 'pixel', 'centered_pixel' 중 하나)
        regression_label (str): 회귀 레이블 유형 ('raw_ret', 'vol_adjust_ret' 또는 None)
        save_dir (str): 데이터 저장 디렉토리
        images (numpy.ndarray): 차트 이미지 데이터
        label_dict (dict): 레이블 데이터 딕셔너리
        demean (list): 정규화를 위한 평균과 표준편차
        ret_val_name (str): 수익률 값 이름
        label (numpy.ndarray): 레이블 데이터

    Methods:
        filter_data: 데이터 필터링
        get_label_value: 레이블 값 계산
        _get_insample_mean_std: 인샘플 평균과 표준편차 계산
        __get_stock_dataset_name: 주식 데이터셋 이름 생성
        get_image_label_save_path: 이미지와 레이블 저장 경로 반환
        rebuild_image: 이미지 재구성
        load_image_np_data: 이미지 데이터 로드
        load_annual_data_by_country: 국가별 연간 데이터 로드
        load_images_and_labels_by_country: 국가별 이미지와 레이블 로드
        __len__: 데이터셋 길이 반환
        __getitem__: 인덱스에 해당하는 샘플 반환
    """
    def __init__(
        self,
        window_size,
        predict_window,
        freq,
        year,
        country="USA",
        has_volume_bar=True,
        has_ma=True,
        chart_type="bar",
        annual_stocks_num="all",
        tstat_threshold=0,
        stockid_filter=None,
        remove_tail=False,
        ohlc_len=None,
        regression_label=None,
        delayed_ret=0,
    ):
        """
        EquityDataset 클래스의 생성자입니다.

        Args:
            window_size (int): 윈도우 크기
            predict_window (int): 예측 윈도우 크기
            freq (str): 데이터 빈도
            year (int): 데이터 연도
            country (str, optional): 국가 코드. 기본값은 "USA"
            has_volume_bar (bool, optional): 거래량 바 포함 여부. 기본값은 True
            has_ma (bool, optional): 이동평균선 포함 여부. 기본값은 True
            chart_type (str, optional): 차트 유형. 기본값은 "bar"
            annual_stocks_num (str, optional): 연간 주식 수. 기본값은 "all"
            tstat_threshold (int, optional): t-통계량 임계값. 기본값은 0
            stockid_filter (list, optional): 주식 ID 필터. 기본값은 None
            remove_tail (bool, optional): 꼬리 제거 여부. 기본값은 False
            ohlc_len (int, optional): OHLC 데이터 길이. 기본값은 None
            regression_label (str, optional): 회귀 레이블 유형. 기본값은 None
            delayed_ret (int, optional): 지연된 수익률. 기본값은 0
        """
        self.ws = window_size
        self.pw = predict_window
        self.freq = freq
        assert self.freq in ["week", "month", "quarter", "year"]
        self.year = year
        self.ohlc_len = ohlc_len if ohlc_len is not None else window_size
        assert self.ohlc_len in [5, 20, 60]
        self.data_freq = self.freq if self.ohlc_len == self.ws else "month"
        self.country = country
        self.has_vb = has_volume_bar
        self.has_ma = has_ma
        self.chart_type = chart_type
        assert self.chart_type in ["bar", "pixel", "centered_pixel"]
        self.regression_label = regression_label
        assert self.regression_label in [None, "raw_ret", "vol_adjust_ret"]

        self.save_dir = op.join(dcf.STOCKS_SAVEPATH, f"stocks_{country}", "dataset_all")

        self.images, self.label_dict = self.load_images_and_labels_by_country(
            self.country
        )

        self.demean = self._get_insample_mean_std()

        assert delayed_ret in [0, 1, 2, 3, 4, 5]
        if self.country == "USA":
            self.ret_val_name = f"Ret_{dcf.FREQ_DICT[self.pw]}" + (
                "" if delayed_ret == 0 else f"_{delayed_ret}delay"
            )
        else:
            self.ret_val_name = f"next_{dcf.FREQ_DICT[self.pw]}_ret_{delayed_ret}delay"
        self.label = self.get_label_value()

        self.filter_data(
            annual_stocks_num, stockid_filter, tstat_threshold, remove_tail
        )

    def filter_data(self, annual_stocks_num, stockid_filter, tstat_threshold, remove_tail):
        """
        데이터를 필터링합니다.

        Args:
            annual_stocks_num (str): 연간 주식 수
            stockid_filter (list): 주식 ID 필터
            tstat_threshold (int): t-통계량 임계값
            remove_tail (bool): 꼬리 제거 여부

        Returns:
            None
        """
        df = pd.DataFrame(
            {
                "StockID": self.label_dict["StockID"],
                "MarketCap": abs(self.label_dict["MarketCap"]),
                "Date": pd.to_datetime([str(t) for t in self.label_dict["Date"]]),
            }
        )
        if annual_stocks_num != "all":
            num_stockid = len(np.unique(df.StockID))
            new_df = df
            period_end_dates = eqd.get_period_end_dates(self.freq)
            for i in range(15):
                date = period_end_dates[
                    (period_end_dates.year == self.year) & (period_end_dates.month == 6)
                ][
                    -i
                ]
                print(date)
                new_df = df[df.Date == date]
                if len(np.unique(new_df.StockID)) > num_stockid / 2:
                    break
            if stockid_filter is not None:
                new_df = new_df[new_df.StockID.isin(stockid_filter)]
            new_df = new_df.sort_values(by=["MarketCap"], ascending=False)
            if len(new_df) > int(annual_stocks_num):
                stockids = new_df.iloc[: int(annual_stocks_num)]["StockID"]
            else:
                stockids = new_df.StockID
            print(
                f"Year {self.year}: select top {annual_stocks_num} stocks ({len(stockids)}/{num_stockid}) stocks for training"
            )
        else:
            stockids = (
                stockid_filter if stockid_filter is not None else np.unique(df.StockID)
            )
        stockid_idx = pd.Series(df.StockID).isin(stockids)

        idx = (
            stockid_idx
            & pd.Series(self.label != -99)
            & pd.Series(self.label_dict["EWMA_vol"] != 0.0)
        )

        if tstat_threshold != 0:
            tstats = np.divide(
                self.label_dict[self.ret_val_name], np.sqrt(self.label_dict["EWMA_vol"])
            )
            tstats = np.abs(tstats)
            t_th = np.nanpercentile(tstats[idx], tstat_threshold)
            tstat_idx = tstats > t_th
            print(
                f"Before filtering bottom {tstat_threshold}% tstats, sample size:{np.sum(idx)}"
            )
            idx = idx & tstat_idx
            print(
                f"After filtering bottom {tstat_threshold}% tstats, sample size:{np.sum(idx)}"
            )

        if remove_tail:
            print(
                f"I{self.ws}R{self.pw}: removing tail for year {self.year} ({np.sum(idx)} samples)"
            )
            last_day = "12/24" if self.pw == 5 else "12/1" if self.pw == 20 else "10/1"
            last_day = pd.Timestamp("{}/{}".format(last_day, self.year))
            idx = idx & (
                pd.to_datetime([str(t) for t in self.label_dict["Date"]]) < last_day
            )

        if self.freq != self.data_freq and self.ohlc_len != self.ws:
            assert self.freq in ["quarter", "year"] and self.data_freq == "month"
            print(f"Selecting data of freq {self.freq}")
            dates = pd.DatetimeIndex(self.label_dict["Date"])
            date_idx = (
                dates.month.isin([3, 6, 9, 12])
                if self.freq == "quarter"
                else dates.month == 12
            )
            idx = idx & date_idx

        self.label = self.label[idx]
        print(f"Year {self.year}: samples size: {len(self.label)}")
        for k in self.label_dict.keys():
            self.label_dict[k] = self.label_dict[k][idx]
        self.images = self.images[idx]
        self.label_dict["StockID"] = self.label_dict["StockID"].astype(str)
        self.label_dict["Date"] = self.label_dict["Date"].astype(str)

        assert len(self.label) == len(self.images)
        for k in self.label_dict.keys():
            assert len(self.images) == len(self.label_dict[k])

    def get_label_value(self):
        """
        레이블 값을 계산합니다.

        Returns:
            numpy.ndarray: 계산된 레이블 값
        """
        print(f"Using {self.ret_val_name} as label")
        ret = self.label_dict[self.ret_val_name]

        print(
            f"Using {self.regression_label} regression label (None represents classification label)"
        )
        if self.regression_label == "raw_ret":
            label = np.nan_to_num(ret, nan=-99)
        elif self.regression_label == "vol_adjust_ret":
            label = np.nan_to_num(ret / np.sqrt(self.label_dict["EWMA_vol"]), nan=-99)
        else:
            label = np.where(ret > 0, 1, 0)
            label = np.nan_to_num(label, nan=-99)

        return label

    def _get_insample_mean_std(self):
        """
        인샘플 평균과 표준편차를 계산합니다.

        Returns:
            list: [평균, 표준편차]
        """
        ohlc_len_srt = f"_{self.ohlc_len}ohlc" if self.ohlc_len != self.ws else ""
        chart_str = f"_{self.chart_type}" if self.chart_type != "bar" else ""
        fname = f"mean_std_{self.ws}d{self.data_freq}_vb{self.has_vb}_ma{self.has_ma}_{self.year}{ohlc_len_srt}{chart_str}.npz"
        mean_std_path = op.join(self.save_dir, fname)
        if op.exists(mean_std_path):
            print(f"Loading mean and std from {mean_std_path}")
            x = np.load(mean_std_path, allow_pickle=True)
            demean = [x["mean"], x["std"]]
            return demean

        print(f"Calculating mean and std for {fname}")
        mean, std = (
            self.images[:50000].mean() / 255.0,
            self.images[:50000].std() / 255.0,
        )
        np.savez(mean_std_path, mean=mean, std=std)
        return [mean, std]

    def __get_stock_dataset_name(self):
        chart_type_str = "" if self.chart_type == "bar" else f"{self.chart_type}_"
        vb_str = "has_vb" if self.has_vb else "no_vb"
        ma_str = f"[{self.ws}]_ma" if self.has_ma else "None_ma"
        str_list = [
            chart_type_str + f"{self.ws}d",
            self.data_freq,
            vb_str,
            ma_str,
            str(self.year),
        ]
        if self.ohlc_len != self.ws:
            str_list.append(f"{self.ohlc_len}ohlc")
        dataset_name = "_".join(str_list)
        return dataset_name

    def get_image_label_save_path(self, country=None):
        if country is None:
            country = self.country
        save_dir = op.join(dcf.STOCKS_SAVEPATH, f"stocks_{country}", "dataset_all")
        dataset_name = self.__get_stock_dataset_name()
        img_save_path = op.join(save_dir, f"{dataset_name}_images.dat")
        label_path = op.join(save_dir, f"{dataset_name}_labels.feather")
        return img_save_path, label_path

    @staticmethod
    def rebuild_image(image, image_name, par_save_dir, image_mode="L"):
        """
        이미지를 재구성하고 저장합니다.

        Args:
            image (numpy.ndarray): 재구성할 이미지 데이터
            image_name (str): 저장할 이미지 이름
            par_save_dir (str): 상위 저장 디렉토리
            image_mode (str, optional): 이미지 모드. 기본값은 "L"

        Returns:
            None
        """
        img = Image.fromarray(image, image_mode)
        save_dir = ut.get_dir(op.join(par_save_dir, "images_rebuilt_from_dataset/"))
        img.save(op.join(save_dir, "{}.png".format(image_name)))

    @staticmethod
    def load_image_np_data(img_save_path, ohlc_len):
        """
        이미지 데이터를 numpy 배열로 로드합니다.

        Args:
            img_save_path (str): 이미지 데이터 저장 경로
            ohlc_len (int): OHLC 데이터 길이

        Returns:
            numpy.ndarray: 로드된 이미지 데이터
        """
        images = np.memmap(img_save_path, dtype=np.uint8, mode="r")
        images = images.reshape(
            (-1, 1, dcf.IMAGE_HEIGHT[ohlc_len], dcf.IMAGE_WIDTH[ohlc_len])
        )
        return images

    def load_annual_data_by_country(self, country):
        """
        국가별 연간 데이터를 로드합니다.

        Args:
            country (str): 국가 코드

        Returns:
            tuple: (이미지 데이터, 레이블 딕셔너리)
        """
        img_save_path, label_path = self.get_image_label_save_path(country)

        print(f"loading images from {img_save_path}")
        images = self.load_image_np_data(img_save_path, self.ohlc_len)
        self.rebuild_image(
            images[0][0],
            image_name=self.__get_stock_dataset_name(),
            par_save_dir=self.save_dir,
        )

        label_df = pd.read_feather(label_path)
        label_df["StockID"] = label_df["StockID"].astype(str)
        label_dict = {c: np.array(label_df[c]) for c in label_df.columns}

        return images, label_dict

    def load_images_and_labels_by_country(self, country):
        """
        국가별 이미지와 레이블을 로드합니다.

        Args:
            country (str): 국가 코드

        Returns:
            tuple: (이미지 데이터, 레이블 딕셔너리)
        """
        images, label_dict = self.load_annual_data_by_country(country)
        return images, label_dict

    def __len__(self):
        """
        데이터셋의 길이를 반환합니다.

        Returns:
            int: 데이터셋의 길이
        """
        return len(self.label)

    def __getitem__(self, idx):
        """
        주어진 인덱스에 해당하는 샘플을 반환합니다.

        Args:
            idx (int): 샘플 인덱스

        Returns:
            dict: 샘플 데이터 (이미지, 레이블, 수익률 값, 종료 날짜, 주식 ID, 시가총액)
        """
        image = (self.images[idx] / 255.0 - self.demean[0]) / self.demean[1]

        sample = {
            "image": image,
            "label": self.label[idx],
            "ret_val": self.label_dict[self.ret_val_name][idx],
            "ending_date": self.label_dict["Date"][idx],
            "StockID": self.label_dict["StockID"][idx],
            "MarketCap": self.label_dict["MarketCap"][idx],
        }
        return sample


def load_ts1d_np_data(ts1d_save_path, ohlc_len):
    """
    1차원 시계열 데이터를 numpy 배열로 로드합니다.

    Args:
        ts1d_save_path (str): 1차원 시계열 데이터 저장 경로
        ohlc_len (int): OHLC 데이터 길이

    Returns:
        numpy.ndarray: 로드된 1차원 시계열 데이터
    """
    images = np.memmap(ts1d_save_path, dtype=np.uint8, mode="r")
    images = images.reshape(
        (-1, 6, dcf.IMAGE_HEIGHT[ohlc_len], dcf.IMAGE_WIDTH[ohlc_len])
    )
    return images


class TS1DDataset(Dataset):
    """
    1차원 시계열 데이터셋을 위한 클래스입니다.
    
    이 클래스는 1차원 시계열 주식 데이터와 관련 레이블을 로드하고 처리합니다.
    PyTorch의 Dataset 클래스를 상속받아 구현되었습니다.

    Attributes:
        ws (int): 윈도우 크기
        pw (int): 예측 윈도우 크기
        freq (str): 데이터 빈도
        year (int): 데이터 연도
        ohlc_len (int): OHLC 데이터 길이
        data_freq (str): 실제 데이터 빈도
        country (str): 국가 코드
        remove_tail (bool): 꼬리 제거 여부
        ts_scale (str): 시계열 스케일링 방법
        regression_label (str): 회귀 레이블 유형
        ret_val_name (str): 수익률 값 이름
        images (numpy.ndarray): 시계열 이미지 데이터
        label_dict (dict): 레이블 데이터 딕셔너리
        label (numpy.ndarray): 레이블 데이터
        demean (list): 정규화를 위한 평균과 표준편차

    Methods:
        load_ts1d_data: 1차원 시계열 데이터 로드
        get_label_value: 레이블 값 계산
        filter_data: 데이터 필터링
        __get_stock_dataset_name: 주식 데이터셋 이름 생성
        _get_1d_mean_std: 1차원 데이터의 평균과 표준편차 계산
        _minmax_scale_ts1d: 1차원 시계열 데이터 Min-Max 스케일링
        _vol_scale_ts1d: 1차원 시계열 데이터 변동성 스케일링
        __len__: 데이터셋 길이 반환
        __getitem__: 인덱스에 해당하는 샘플 반환
    """
    def __init__(
        self,
        window_size,
        predict_window,
        freq,
        year,
        country="USA",
        remove_tail=False,
        ohlc_len=None,
        ts_scale="image_scale",
        regression_label=None,
    ):
        """
        TS1DDataset 클래스의 생성자입니다.

        Args:
            window_size (int): 윈도우 크기
            predict_window (int): 예측 윈도우 크기
            freq (str): 데이터 빈도
            year (int): 데이터 연도
            country (str, optional): 국가 코드. 기본값은 "USA"
            remove_tail (bool, optional): 꼬리 제거 여부. 기본값은 False
            ohlc_len (int, optional): OHLC 데이터 길이. 기본값은 None
            ts_scale (str, optional): 시계열 스케일링 방법. 기본값은 "image_scale"
            regression_label (str, optional): 회귀 레이블 유형. 기본값은 None
        """
        self.ws = window_size
        self.pw = predict_window
        self.freq = freq
        self.year = year
        self.ohlc_len = ohlc_len if ohlc_len is not None else window_size
        self.data_freq = self.freq if self.ohlc_len == self.ws else "month"
        self.country = country
        self.remove_tail = remove_tail
        self.ts_scale = ts_scale
        assert self.ts_scale in ["image_scale", "ret_scale", "vol_scale"]
        self.regression_label = regression_label
        assert self.regression_label in [None, "raw_ret", "vol_adjust_ret"]
        self.ret_val_name = f"Retx_{dcf.FREQ_DICT[self.pw]}"
        self.images, self.label_dict = self.load_ts1d_data()
        self.label = self.get_label_value()
        self.demean = self._get_1d_mean_std()
        self.filter_data(self.remove_tail)

    def load_ts1d_data(self):
        """
        1차원 시계열 데이터를 로드합니다.

        Returns:
            tuple: (이미지 데이터, 레이블 딕셔너리)
        """
        dataset_name = self.__get_stock_dataset_name()
        filename = op.join(
            dcf.STOCKS_SAVEPATH,
            "stocks_USA_ts/dataset_all/",
            "{}_data_new.npz".format(dataset_name),
        )
        data = np.load(filename, mmap_mode="r", encoding="latin1", allow_pickle=True)
        label_dict = data["data_dict"].item()
        images = label_dict["predictor"].copy()
        assert images[0].shape == (6, self.ohlc_len)
        del label_dict["predictor"]
        label_dict["StockID"] = label_dict["StockID"].astype(str)
        return images, label_dict

    def get_label_value(self):
        """
        레이블 값을 계산합니다.

        Returns:
            numpy.ndarray: 계산된 레이블 값
        """
        print(f"Using {self.ret_val_name} as label")
        ret = self.label_dict[self.ret_val_name]

        print(
            f"Using {self.regression_label} regression label (None represents classification label)"
        )
        if self.regression_label == "raw_ret":
            label = np.nan_to_num(ret, nan=-99)
        elif self.regression_label == "vol_adjust_ret":
            label = np.nan_to_num(ret / np.sqrt(self.label_dict["EWMA_vol"]), nan=-99)
        else:
            label = np.where(ret > 0, 1, 0)
            label = np.nan_to_num(label, nan=-99)
        return label

    def filter_data(self, remove_tail):
        """
        데이터를 필터링합니다.

        Args:
            remove_tail (bool): 꼬리 제거 여부

        Returns:
            None
        """
        idx = pd.Series(self.label != -99) & pd.Series(
            self.label_dict["EWMA_vol"] != 0.0
        )

        if remove_tail:
            print(
                f"I{self.ws}R{self.pw}: removing tail for year {self.year} ({np.sum(idx)} samples)"
            )
            last_day = "12/24" if self.pw == 5 else "12/1" if self.pw == 20 else "10/1"
            last_day = pd.Timestamp("{}/{}".format(last_day, self.year))
            idx = idx & (
                pd.to_datetime([str(t) for t in self.label_dict["Date"]]) < last_day
            )

        if self.freq != self.data_freq and self.ohlc_len != self.ws:
            assert self.freq in ["quarter", "year"] and self.data_freq == "month"
            print(f"Selecting data of freq {self.freq}")
            dates = pd.DatetimeIndex(self.label_dict["Date"])
            date_idx = (
                dates.month.isin([3, 6, 9, 12])
                if self.freq == "quarter"
                else dates.month == 12
            )
            idx = idx & date_idx

        self.label = self.label[idx]
        print(f"Year {self.year}: samples size: {len(self.label)}")
        for k in self.label_dict.keys():
            self.label_dict[k] = self.label_dict[k][idx]
        self.images = self.images[idx]
        self.label_dict["StockID"] = self.label_dict["StockID"].astype(str)
        self.label_dict["Date"] = self.label_dict["Date"].astype(str)

        assert len(self.label) == len(self.images)
        for k in self.label_dict.keys():
            assert len(self.images) == len(self.label_dict[k])

    def __get_stock_dataset_name(self):
        """
        주식 데이터셋 이름을 생성합니다.

        Returns:
            str: 생성된 데이터셋 이름
        """
        str_list = [
            f"{self.ws}d",
            self.data_freq,
            "has_vb",
            f"[{self.ws}]_ma",
            str(self.year),
        ]
        if self.ohlc_len != self.ws:
            str_list.append(f"{self.ohlc_len}ohlc")
        str_list.append("ts")
        dataset_name = "_".join(str_list)
        return dataset_name

    def _get_1d_mean_std(self):
        ohlc_len_srt = f"_{self.ohlc_len}ohlc" if self.ohlc_len != self.ws else ""
        raw_surfix = (
            ""
            if self.ts_scale == "image_scale"
            else "_raw_price"
            if self.ts_scale == "ret_scale"
            else "_vol_scale"
        )
        fname = f"mean_std_ts1d_{self.ws}d{self.data_freq}_vbTrue_maTrue_{self.year}{ohlc_len_srt}{raw_surfix}.npz"
        mean_std_path = op.join(
            ut.get_dir(
                op.join(dcf.STOCKS_SAVEPATH, f"stocks_{self.country}_ts", "dataset_all")
            ),
            fname,
        )

        if op.exists(mean_std_path):
            print(f"Loading mean and std from {mean_std_path}")
            x = np.load(mean_std_path, allow_pickle=True)
            demean = [x["mean"], x["std"]]
            return demean

        if self.ts_scale == "image_scale":
            for i in range(self.images.shape[0]):
                self.images[i] = self._minmax_scale_ts1d(self.images[i])
        elif self.ts_scale == "vol_scale":
            for i in range(self.images.shape[0]):
                self.images[i] = self._vol_scale_ts1d(self.images[i]) / np.sqrt(
                    self.label_dict["EWMA_vol"][i]
                )

        print(f"Calculating mean and std for {fname}")
        mean, std = np.nanmean(self.images, axis=(0, 2)), np.nanstd(
            self.images, axis=(0, 2)
        )
        np.savez(mean_std_path, mean=mean, std=std)
        return [mean, std]

    def _minmax_scale_ts1d(self, image):
        assert image.shape == (6, self.ohlc_len)
        ohlcma = image[:5]
        image[:5] = (ohlcma - np.nanmin(ohlcma)) / (
            np.nanmax(ohlcma) - np.nanmin(ohlcma)
        )
        image[5] = (image[5] - np.nanmin(image[5])) / (
            np.nanmax(image[5]) - np.nanmin(image[5])
        )
        return image

    def _vol_scale_ts1d(self, image):
        img = image.copy()
        img[:, 0] = 0
        for i in range(1, 5):
            img[:, i] = image[:, i] / image[0, i - 1] - 1
        return img

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.ts_scale == "image_scale":
            image = self._minmax_scale_ts1d(image)
        elif self.ts_scale == "vol_scale":
            image = self._vol_scale_ts1d(image) / np.sqrt(
                self.label_dict["EWMA_vol"][idx]
            )

        image = (image - self.demean[0].reshape(6, 1)) / self.demean[1].reshape(6, 1)
        image = np.nan_to_num(image, nan=0, posinf=0, neginf=0)

        sample = {
            "image": image,
            "label": self.label[idx],
            "ret_val": self.label_dict[self.ret_val_name][idx],
            "ending_date": self.label_dict["Date"][idx],
            "StockID": self.label_dict["StockID"][idx],
            "MarketCap": self.label_dict["MarketCap"][idx],
        }
        return sample


def main():
    pass


if __name__ == "__main__":
    main()
