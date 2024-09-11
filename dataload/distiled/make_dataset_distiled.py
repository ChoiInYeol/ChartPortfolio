import json
import pickle
import numpy as np
import pandas as pd


def get_return_df(stock_dic, n_stock, in_path="data/stocks/", out_path="data/"):
    return_df = pd.DataFrame()
    for ticker in list(stock_dic.keys())[:n_stock]:
        stock = pd.read_csv(in_path + f"{ticker}.csv", index_col="Date", parse_dates=True)["Adj Close"]
        return_df[ticker] = np.log(stock) - np.log(stock.shift(1))
    return_df = return_df.dropna()
    return_df.to_csv(out_path + "return_df.csv")
    return return_df


def make_DL_dataset(data, data_len, n_stock):
    times = []
    dataset = np.array(data.iloc[:data_len, :]).reshape(1, -1, n_stock)
    times.append(data.iloc[:data_len, :].index)

    for i in range(1, len(data) - data_len + 1):
        addition = np.array(data.iloc[i : data_len + i, :]).reshape(1, -1, n_stock)
        dataset = np.concatenate((dataset, addition))
        times.append(data.iloc[i : data_len + i, :].index)
    return dataset, times


def data_split(data, train_len, pred_len, tr_ratio, n_stock):
    split_index = int(len(data) * tr_ratio)
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    x_tr, y_tr, times_tr = [], [], []
    for i in range(len(train_data) - train_len - pred_len + 1):
        x_tr.append(train_data.iloc[i:i+train_len, :n_stock].values)
        y_tr.append(train_data.iloc[i+train_len:i+train_len+pred_len, :n_stock].values)
        times_tr.append(train_data.index[i+train_len+pred_len-1])

    x_te, y_te, times_te = [], [], []
    for i in range(len(test_data) - train_len - pred_len + 1):
        x_te.append(test_data.iloc[i:i+train_len, :n_stock].values)
        y_te.append(test_data.iloc[i+train_len:i+train_len+pred_len, :n_stock].values)
        times_te.append(test_data.index[i+train_len+pred_len-1])

    return np.array(x_tr), np.array(y_tr), np.array(x_te), np.array(y_te), times_tr, times_te

def test_dataset():
    path = "data/"
    config = json.load(open("config/data_config.json", "r", encoding="utf8"))
    stock_dict_sp = json.load(open(path + "stock.json", "r", encoding="UTF8"))
    return_df = get_return_df(stock_dict_sp, config["N_STOCK"])
    x_tr, y_tr, x_te, y_te, times_tr, times_te = data_split(
        return_df,
        config["TRAIN_LEN"],
        config["PRED_LEN"],
        config["TRAIN_RATIO"],
        config["N_STOCK"],
    )

    print("데이터셋 검증:")
    print(f"x_tr shape: {x_tr.shape}")
    print(f"y_tr shape: {y_tr.shape}")
    print(f"x_te shape: {x_te.shape}")
    print(f"y_te shape: {y_te.shape}")
    print(f"times_tr length: {len(times_tr)}")
    print(f"times_te length: {len(times_te)}")

    assert x_tr.shape[1] == config["TRAIN_LEN"], "x_tr의 시퀀스 길이가 올바르지 않습니다."
    assert x_tr.shape[2] == config["N_STOCK"], "x_tr의 주식 수가 올바르지 않습니다."
    assert y_tr.shape[1] == config["PRED_LEN"], "y_tr의 예측 길이가 올바르지 않습니다."
    assert y_tr.shape[2] == config["N_STOCK"], "y_tr의 주식 수가 올바르지 않습니다."
    assert x_te.shape[1] == config["TRAIN_LEN"], "x_te의 시퀀스 길이가 올바르지 않습니다."
    assert x_te.shape[2] == config["N_STOCK"], "x_te의 주식 수가 올바르지 않습니다."
    assert y_te.shape[1] == config["PRED_LEN"], "y_te의 예측 길이가 올바르지 않습니다."
    assert y_te.shape[2] == config["N_STOCK"], "y_te의 주식 수가 올바르지 않습니다."

    print("모든 검증 통과")

if __name__ == "__main__":
    path = "data/"
    config = json.load(open("config/data_config.json", "r", encoding="utf8"))
    stock_dict_sp = json.load(open(path + "stock.json", "r", encoding="UTF8"))
    return_df = get_return_df(stock_dict_sp, config["N_STOCK"])
    x_tr, y_tr, x_te, y_te, times_tr, times_te = data_split(
        return_df,
        config["TRAIN_LEN"],
        config["PRED_LEN"],
        config["TRAIN_RATIO"],
        config["N_STOCK"],
    )

    with open(path + "date.pkl", "wb") as f:
        pickle.dump({'train': times_tr, 'test': times_te}, f)

    with open(path + "dataset.pkl", "wb") as f:
        pickle.dump([x_tr, y_tr, x_te, y_te], f)
    
    test_dataset()
