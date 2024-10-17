import yaml
import numpy as np
import pandas as pd
import logging
import os
import gc
import h5py
import pickle

logging.basicConfig(filename='make_dataset.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_ticker_mapping(file_path="data/ticker_stockid.csv"):
    ticker_mapping = {}
    df = pd.read_csv(file_path)
    for _, row in df.iterrows():
        ticker_mapping[row['TICKER']] = row['StockID']
    return ticker_mapping

def get_return_df_generator(stock_dic, start_date, end_date, in_path="data/filtered_stock.csv", chunk_size=50000):
    reader = pd.read_csv(in_path, usecols=['date', 'PERMNO', 'return'], chunksize=chunk_size, parse_dates=['date'])
    for chunk in reader:
        # 날짜 범위 필터링
        chunk = chunk[(chunk['date'] >= start_date) & (chunk['date'] <= end_date)]
        if chunk.empty:
            del chunk
            gc.collect()
            continue
        # 필요한 종목만 필터링
        chunk_filtered = chunk[chunk['PERMNO'].isin(stock_dic.values())]
        if chunk_filtered.empty:
            del chunk, chunk_filtered
            gc.collect()
            continue
        chunk_filtered.set_index(['date', 'PERMNO'], inplace=True)
        return_df_chunk = chunk_filtered['return'].unstack(level='PERMNO')
        return_df_chunk = return_df_chunk.sort_index()
        return_df_chunk = return_df_chunk.fillna(-2)
        yield return_df_chunk.astype('float32')
        del chunk, chunk_filtered, return_df_chunk
        gc.collect()

def make_DL_dataset_hdf5(data_generator, train_len, pred_len, output_file):
    times = []
    with h5py.File(output_file, 'w') as h5f:
        x_dset = None
        y_dset = None
        count = 0

        buffer = []
        for data_chunk in data_generator:
            buffer.append(data_chunk)
            combined_data = pd.concat(buffer)
            num_rows = combined_data.shape[0]

            while num_rows >= train_len + pred_len:
                x_subset = combined_data.iloc[:train_len].values
                y_subset = combined_data.iloc[train_len:train_len+pred_len].values
                
                # 고정된 열 크기 설정
                if x_dset is None:
                    fixed_shape = x_subset.shape[1]  # 첫 번째 배치의 열 크기로 고정
                    x_dset = h5f.create_dataset('x', data=x_subset[np.newaxis, :], maxshape=(None, train_len, fixed_shape), chunks=True)
                    y_dset = h5f.create_dataset('y', data=y_subset[np.newaxis, :], maxshape=(None, pred_len, fixed_shape), chunks=True)
                else:
                    # 현재 배치의 열 크기가 고정된 크기와 다르면 크기를 맞춤
                    if x_subset.shape[1] != fixed_shape:
                        if x_subset.shape[1] < fixed_shape:
                            # 부족한 부분을 0으로 채움
                            padding = np.zeros((x_subset.shape[0], fixed_shape - x_subset.shape[1]))
                            x_subset = np.hstack((x_subset, padding))
                            y_subset = np.hstack((y_subset, np.zeros((y_subset.shape[0], fixed_shape - y_subset.shape[1]))))
                        else:
                            # 넘치는 부분을 잘라냄
                            x_subset = x_subset[:, :fixed_shape]
                            y_subset = y_subset[:, :fixed_shape]

                    x_dset.resize(x_dset.shape[0] + 1, axis=0)
                    y_dset.resize(y_dset.shape[0] + 1, axis=0)
                    x_dset[-1] = x_subset
                    y_dset[-1] = y_subset
                
                times.append(combined_data.index[train_len+pred_len-1])
                combined_data = combined_data.iloc[1:]
                num_rows = combined_data.shape[0]

            buffer = [combined_data]
            del data_chunk
            gc.collect()

    return times


if __name__ == "__main__":
    path = "data/"
    with open("config/config.yaml", "r", encoding="utf8") as file:
        config = yaml.safe_load(file)

    stock_dict_sp = load_ticker_mapping(path + "ticker_stockid.csv")

    # 날짜 설정
    train_start_date = config["TRAIN_START_DATE"]
    train_end_date = config["TRAIN_END_DATE"]
    validation_ratio = config.get("VALIDATION_RATIO", 0.2)
    total_days = (pd.to_datetime(train_end_date) - pd.to_datetime(train_start_date)).days
    validation_days = int(total_days * validation_ratio)
    validation_start_date = (pd.to_datetime(train_end_date) - pd.Timedelta(days=validation_days)).strftime('%Y-%m-%d')

    test_start_date = config["TEST_START_DATE"]
    test_end_date = config["TEST_END_DATE"]

    # 제너레이터 생성
    train_data_generator = get_return_df_generator(stock_dict_sp, train_start_date, validation_start_date)
    validation_data_generator = get_return_df_generator(stock_dict_sp, validation_start_date, train_end_date)
    test_data_generator = get_return_df_generator(stock_dict_sp, test_start_date, test_end_date)

    # HDF5로 데이터셋 저장
    train_times = make_DL_dataset_hdf5(train_data_generator, config["TRAIN_LEN"], config["PRED_LEN"], path + "train_dataset.h5")
    validation_times = make_DL_dataset_hdf5(validation_data_generator, config["TRAIN_LEN"], config["PRED_LEN"], path + "validation_dataset.h5")
    test_times = make_DL_dataset_hdf5(test_data_generator, config["TRAIN_LEN"], config["PRED_LEN"], path + "test_dataset.h5")

    # 시간 정보 저장
    with open(path + "train_times.pkl", 'wb') as f:
        pickle.dump(train_times, f)
    with open(path + "validation_times.pkl", 'wb') as f:
        pickle.dump(validation_times, f)
    with open(path + "test_times.pkl", 'wb') as f:
        pickle.dump(test_times, f)

    logging.info("데이터셋 생성 완료")
    print("Data preparation is completed.")
