import yaml
import pickle
import numpy as np
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# 로깅 설정
logging.basicConfig(filename='make_dataset.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def process_stock(ticker, in_path):
    """
    개별 주식 데이터를 처리합니다.

    Args:
        ticker (str): 주식 심볼
        in_path (str): 입력 데이터 경로

    Returns:
        pd.Series: 처리된 주식 데이터
    """
    try:
        stock = pd.read_csv(in_path + f"{ticker}.csv", index_col="Date", parse_dates=True)
        stock_return = np.log(stock["Adj Close"]) - np.log(stock["Adj Close"].shift(1))
        return stock_return.rename(ticker)
    except Exception as e:
        logging.warning(f"Error processing {ticker}: {e}")
        return None

def get_return_df(stock_dic, in_path="data/stocks/", out_path="data/"):
    """
    주식 데이터를 읽고 로그 수익률을 계산합니다.

    Args:
        stock_dic (dict): 주식 심볼 딕셔너리
        in_path (str): 입력 데이터 경로
        out_path (str): 출력 데이터 경로

    Returns:
        pd.DataFrame: 로그 수익률 데이터프레임
    """
    stock_returns = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_stock, ticker, in_path): ticker for ticker in stock_dic}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                stock_returns.append(result)
    
    return_df = pd.concat(stock_returns, axis=1)
    return_df = return_df.dropna(how='all')  # 모든 값이 NaN인 행 제거
    return_df = return_df.fillna(-2)  # 나머지 NaN을 -2으로 채움
    return_df.to_csv(out_path + "return_df.csv")
    return return_df

def make_DL_dataset(data, data_len):
    """
    딥러닝 데이터셋을 생성합니다.

    Args:
        data (pd.DataFrame): 입력 데이터
        data_len (int): 데이터 길이

    Returns:
        tuple: (dataset, times)
    """
    times = []
    dataset = []
    for i in range(len(data) - data_len + 1):
        subset = data.iloc[i:i+data_len]
        dataset.append(subset.values)
        times.append(subset.index)
    return np.array(dataset), times

def data_split(data, train_len, pred_len, train_ratio, val_ratio):
    """
    데이터를 훈련, 검증, 테스트 세트로 분할합니다.

    Args:
        data (pd.DataFrame): 입력 데이터
        train_len (int): 훈련 데이터 길이
        pred_len (int): 예측 길이
        train_ratio (float): 훈련 데이터 비율
        val_ratio (float): 검증 데이터 비율

    Returns:
        tuple: (x_tr, y_tr, x_val, y_val, x_te, y_te, times_tr, times_val, times_te)
    """
    total_len = len(data)
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))

    return_train, times_train = make_DL_dataset(data[:train_end], train_len + pred_len)
    return_val, times_val = make_DL_dataset(data[train_end:val_end], train_len + pred_len)
    return_test, times_test = make_DL_dataset(data[val_end:], train_len + pred_len)

    x_tr, y_tr = return_train[:, :train_len], return_train[:, -pred_len:]
    x_val, y_val = return_val[:, :train_len], return_val[:, -pred_len:]
    x_te, y_te = return_test[:, :train_len], return_test[:, -pred_len:]

    times_tr = [t[-pred_len:] for t in times_train]
    times_val = [t[-pred_len:] for t in times_val]
    times_te = [t[-pred_len:] for t in times_test]

    return x_tr, y_tr, x_val, y_val, x_te, y_te, times_tr, times_val, times_te

if __name__ == "__main__":
    path = "data/"
    with open("config/config.yaml", "r", encoding="utf8") as file:
        config = yaml.safe_load(file)
    
    with open(path + "stock.yaml", "r", encoding="UTF8") as f:
        stock_dict_sp = yaml.safe_load(f)
    
    return_df = get_return_df(stock_dict_sp)
    x_tr, y_tr, x_val, y_val, x_te, y_te, times_tr, times_val, times_te = data_split(
        return_df,
        config["TRAIN_LEN"],
        config["PRED_LEN"],
        config["TRAIN_RATIO"],
        config["VAL_RATIO"]
    )
    
    logging.info("데이터셋 검증:")
    logging.info(f"Train images shape: {x_tr.shape}")
    logging.info(f"Train labels shape: {y_tr.shape}")
    logging.info(f"Validation images shape: {x_val.shape}")
    logging.info(f"Validation labels shape: {y_val.shape}")
    logging.info(f"Test images shape: {x_te.shape}")
    logging.info(f"Test labels shape: {y_te.shape}")
    logging.info(f"Train times length: {len(times_tr)}, Train times last: {times_tr[-1][-1]}")
    logging.info(f"Validation times length: {len(times_val)}, Validation times last: {times_val[-1][-1]}")
    logging.info(f"Test times length: {len(times_te)}, Test times last: {times_te[-1][-1]}")

    with open("data/date.pkl", "wb") as f:
        pickle.dump({'train': times_tr, 'val': times_val, 'test': times_te}, f)

    with open(path + "dataset.pkl", "wb") as f:
        pickle.dump([x_tr, y_tr, x_val, y_val, x_te, y_te], f)

    logging.info("데이터셋 생성 완료")