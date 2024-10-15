import yaml
import pickle
import numpy as np
import pandas as pd
import logging

# 로깅 설정
logging.basicConfig(filename='make_dataset.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def make_DL_dataset(data, data_len):
    """
    딥러닝 데이터셋을 생성합니다.

    Args:
        data (pd.DataFrame): 입력 데이터
        data_len (int): 데이터 길이

    Returns:
        tuple: (dataset, times)
    """
    dataset = []
    times = []

    for i in range(len(data) - data_len + 1):
        subset = data.iloc[i:i + data_len]
        dataset.append(subset.values)  # numpy array로 변환
        times.append(subset.index)

    return np.array(dataset), np.array(times)  # numpy 배열로 반환

def data_split_by_date(data, train_len, pred_len, train_start, train_end, 
                       val_ratio, batch_size=1000):
    """
    날짜와 비율을 사용해 데이터를 훈련 및 검증으로 분할합니다.

    Args:
        data (pd.DataFrame): 입력 데이터
        train_len (int): 훈련 데이터 길이
        pred_len (int): 예측 데이터 길이
        train_start (str): 훈련 시작 날짜
        train_end (str): 훈련 종료 날짜
        val_ratio (float): 검증 세트 비율
        batch_size (int): 배치 크기

    Returns:
        tuple: (x_tr, y_tr, x_val, y_val, x_te, y_te)
    """
    # 날짜 범위에 따른 데이터 선택
    train_data = data.loc[train_start:train_end]

    # 훈련 및 검증 데이터 비율로 분할
    train_size = int(len(train_data) * (1 - val_ratio))
    
    data_train = train_data.iloc[:train_size]
    data_val = train_data.iloc[train_size:]

    # 테스트 데이터 설정
    data_test = data.loc[train_end:]

    # 데이터셋 생성
    x_tr, times_tr = make_DL_dataset(data_train, train_len + pred_len)
    x_val, times_val = make_DL_dataset(data_val, train_len + pred_len)
    x_te, times_te = make_DL_dataset(data_test, train_len + pred_len)

    # 입력과 라벨 분리
    y_tr = x_tr[:, -pred_len:]
    x_tr = x_tr[:, :train_len]

    y_val = x_val[:, -pred_len:]
    x_val = x_val[:, :train_len]

    y_te = x_te[:, -pred_len:]
    x_te = x_te[:, :train_len]

    return x_tr, y_tr, x_val, y_val, x_te, y_te, times_tr, times_val, times_te

if __name__ == "__main__":
    path = "data/"

    # 설정 파일 로드
    with open("config/config.yaml", "r", encoding="utf8") as file:
        config = yaml.safe_load(file)

    # 데이터 로드
    df = pd.read_csv("data/data.csv", index_col="Date", parse_dates=True)

    # NaN 값 처리 및 float32로 변환
    data = df.fillna(-2).astype(np.float32)

    # 데이터 분할
    x_tr, y_tr, x_val, y_val, x_te, y_te, times_tr, times_val, times_te = data_split_by_date(
        data,
        config["TRAIN_LEN"],
        config["PRED_LEN"],
        train_start=config["TRAIN_START_DATE"],
        train_end=config["VAL_END_DATE"],
        val_ratio=0.20,  # 검증 비율 설정
        batch_size=1000
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
    with open("data/date.pkl", "wb") as f:
        pickle.dump({'train': times_tr, 'val': times_val, 'test': times_te}, f)

    # 데이터셋 저장
    with open(path + "dataset.pkl", "wb") as f:
        pickle.dump([x_tr, y_tr, x_val, y_val, x_te, y_te], f)

    logging.info("데이터셋 생성 완료")
