import yaml
import pickle
import numpy as np
import pandas as pd
import logging

# 로깅 설정
logging.basicConfig(filename='make_dataset.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_return_df(filtered_stock_path, ticker_stockid_path):
    """
    필터링된 주식 데이터와 티커-PERMNO 매핑을 사용하여 수익률 데이터프레임을 생성합니다.

    Args:
        filtered_stock_path (str): 필터링된 주식 데이터 파일 경로
        ticker_stockid_path (str): 티커-PERMNO 매핑 파일 경로

    Returns:
        pd.DataFrame: 피벗된 수익률 데이터프레임
    """
    filtered_stock = pd.read_csv(filtered_stock_path, parse_dates=['date'])
    filtered_stock = filtered_stock[['date', 'PERMNO', 'RET']]

    ticker_stockid = pd.read_csv(ticker_stockid_path)
    ticker_stockid = ticker_stockid[['TICKER', 'StockID']]
    ticker_stockid = ticker_stockid.rename(columns={'StockID': 'PERMNO'})

    merged_data = pd.merge(filtered_stock, ticker_stockid, on='PERMNO', how='inner')
    pivot_data = merged_data.pivot(index='date', columns='TICKER', values='RET')

    return pivot_data

def make_DL_dataset(data, data_len, n_stock):
    """
    딥러닝 데이터셋을 생성합니다.

    Args:
        data (pd.DataFrame): 입력 데이터
        data_len (int): 데이터 길이
        n_stock (int): 주식 수

    Returns:
        tuple: (dataset, times)
    """
    times = []
    dataset = np.array(data.iloc[:data_len, :n_stock]).reshape(1, -1, n_stock)
    times.append(data.iloc[:data_len, :].index)

    for i in range(1, len(data) - data_len + 1):
        addition = np.array(data.iloc[i:data_len+i, :n_stock]).reshape(1, -1, n_stock)
        dataset = np.concatenate((dataset, addition))
        times.append(data.iloc[i:data_len+i, :].index)
    return dataset, times

def data_split(data, train_len, pred_len, train_start, train_end, val_ratio, test_start, test_end, n_stock):
    """
    데이터를 훈련, 검증, 테스트 세트로 분할합니다.

    Args:
        data (pd.DataFrame): 입력 데이터
        train_len (int): 훈련 데이터 길이
        pred_len (int): 예측 데이터 길이
        train_start (str): 훈련 시작 날짜
        train_end (str): 훈련 종료 날짜
        val_ratio (float): 검증 세트 비율
        test_start (str): 테스트 시작 날짜
        test_end (str): 테스트 종료 날짜
        n_stock (int): 주식 수

    Returns:
        tuple: (x_tr, y_tr, x_val, y_val, x_te, y_te, times_tr, times_val, times_te)
    """
    # 훈련 데이터 선택
    train_data = data.loc[train_start:train_end]
    
    # 훈련 및 검증 데이터 분할
    train_size = int(len(train_data) * (1 - val_ratio))
    train_data, val_data = train_data.iloc[:train_size], train_data.iloc[train_size:]
    
    # 테스트 데이터 선택 (OOS)
    test_data = data.loc[test_start:test_end]

    # 데이터셋 생성
    return_train, times_train = make_DL_dataset(train_data, train_len + pred_len, n_stock)
    return_val, times_val = make_DL_dataset(val_data, train_len + pred_len, n_stock)
    return_test, times_test = make_DL_dataset(test_data, train_len + pred_len, n_stock)

    # 입력과 라벨 분리
    x_tr = np.array([x[:train_len] for x in return_train])
    y_tr = np.array([x[-pred_len:] for x in return_train])
    times_tr = np.unique(np.array([x[-pred_len:] for x in times_train]).flatten()).tolist()

    x_val = np.array([x[:train_len] for x in return_val])
    y_val = np.array([x[-pred_len:] for x in return_val])
    times_val = np.unique(np.array([x[-pred_len:] for x in times_val]).flatten()).tolist()

    x_te = np.array([x[:train_len] for x in return_test])
    y_te = np.array([x[-pred_len:] for x in return_test])
    times_te = np.unique(np.array([x[-pred_len:] for x in times_test]).flatten()).tolist()

    return x_tr, y_tr, x_val, y_val, x_te, y_te, times_tr, times_val, times_te

if __name__ == "__main__":
    path = "data/"

    # 설정 파일 로드
    with open("config/config.yaml", "r", encoding="utf8") as file:
        config = yaml.safe_load(file)

    # 수익률 데이터프레임 생성
    return_df = get_return_df(path + "filtered_stock.csv", path + "ticker_stockid.csv")

    # NaN 값 처리 및 float32로 변환
    data = return_df.fillna(-2).astype(np.float32)

    # 데이터 분할
    x_tr, y_tr, x_val, y_val, x_te, y_te, times_tr, times_val, times_te = data_split(
        data,
        config["TRAIN_LEN"],
        config["PRED_LEN"],
        config["TRAIN_START_DATE"],
        config["TRAIN_END_DATE"],
        config["VALIDATION_RATIO"],
        config["TEST_START_DATE"],
        config["TEST_END_DATE"],
        config["N_STOCK"],
    )

    # 데이터셋 정보 로깅
    logging.info("데이터셋 검증:")
    logging.info(f"Train images shape: {x_tr.shape}")
    logging.info(f"Train labels shape: {y_tr.shape}")
    logging.info(f"Validation images shape: {x_val.shape}")
    logging.info(f"Validation labels shape: {y_val.shape}")
    logging.info(f"Test images shape: {x_te.shape}")
    logging.info(f"Test labels shape: {y_te.shape}")

    # 날짜 정보 저장
    with open(path + "date.pkl", "wb") as f:
        pickle.dump({'train': times_tr, 'val': times_val, 'test': times_te}, f)

    # 데이터셋 저장
    with open(path + "dataset.pkl", "wb") as f:
        pickle.dump([x_tr, y_tr, x_val, y_val, x_te, y_te], f)

    logging.info("데이터셋 생성 완료")