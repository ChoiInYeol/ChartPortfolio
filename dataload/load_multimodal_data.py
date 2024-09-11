import numpy as np
import pandas as pd
import pickle
import os
import json

def load_config(config_path="config/data_config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_numerical_data(path="data/", config=None):
    with open(path + "dataset.pkl", "rb") as f:
        x_tr, y_tr, x_te, y_te = pickle.load(f)
    
    with open(path + "date.pkl", "rb") as f:
        dates = pickle.load(f)
        times_tr, times_te = dates['train'], dates['test']
    
    if config:
        train_len = config['TRAIN_LEN']
        n_stocks = config['N_STOCK']
        x_tr = x_tr[:, -train_len:, :n_stocks]
        y_tr = y_tr[:, -train_len:, :n_stocks]
        x_te = x_te[:, :, :n_stocks]
        y_te = y_te[:, :, :n_stocks]
    
    return x_tr, y_tr, x_te, y_te, times_tr, times_te

def load_image_data(chart_dir="data/charts/", config=None):
    all_images = {}
    all_meta_data = {}
    for file in os.listdir(chart_dir):
        if file.endswith(".npz"):
            ticker = file.split("_")[2]
            data = np.load(os.path.join(chart_dir, file), allow_pickle=True)
            all_images[ticker] = data['images']
            all_meta_data[ticker] = data['meta_data']
    return all_images, all_meta_data

def align_data(numerical_data, image_data, meta_data, times, config):
    aligned_data = []
    train_len = config['TRAIN_LEN']
    n_stocks = config['N_STOCK']
    for i, end_date in enumerate(times):
        window_data = numerical_data[i]
        img_data = []
        for ticker_index, ticker in enumerate(list(image_data.keys())[:n_stocks]):
            img_index = next((idx for idx, meta in enumerate(meta_data[ticker]) if pd.Timestamp(meta['End_Date']) <= end_date), None)
            if img_index is not None:
                img_data.append(image_data[ticker][img_index])
            else:
                img_data.append(np.zeros((64, 117)))  # 이미지 데이터가 없는 경우 빈 이미지로 대체
        aligned_data.append((window_data, np.array(img_data)))
    return aligned_data

def load_multimodal_data():
    config = load_config()
    x_tr, y_tr, x_te, y_te, times_tr, times_te = load_numerical_data(config=config)
    all_images, all_meta_data = load_image_data(config=config)
    
    aligned_train = align_data(x_tr, all_images, all_meta_data, times_tr, config)
    aligned_test = align_data(x_te, all_images, all_meta_data, times_te, config)
    
    return aligned_train, aligned_test, times_te, config

def test_multimodal_data():
    config = load_config()
    x_tr, y_tr, x_te, y_te, times_tr, times_te = load_numerical_data(config=config)
    all_images, all_meta_data = load_image_data(config=config)
    
    aligned_train = align_data(x_tr, all_images, all_meta_data, times_tr, config)
    aligned_test = align_data(x_te, all_images, all_meta_data, times_te, config)

    print("멀티모달 데이터 검증:")
    print(f"Aligned train data: {len(aligned_train)}")
    print(f"Aligned test data: {len(aligned_test)}")

    for i, (num_data, img_data) in enumerate(aligned_train):
        assert num_data.shape == (config['TRAIN_LEN'], config['N_STOCK']), f"훈련 데이터 {i}의 수치 데이터 형태가 올바르지 않습니다."
        assert img_data.shape == (config['N_STOCK'], 64, 117), f"훈련 데이터 {i}의 이미지 데이터 형태가 올바르지 않습니다."
        assert img_data.dtype == np.uint8, f"훈련 데이터 {i}의 이미지 데이터 타입이 올바르지 않습니다."
        assert np.min(img_data) >= 0 and np.max(img_data) <= 255, f"훈련 데이터 {i}의 이미지 데이터 값 범위가 올바르지 않습니다."

    for i, (num_data, img_data) in enumerate(aligned_test):
        assert num_data.shape == (config['TRAIN_LEN'], config['N_STOCK']), f"테스트 데이터 {i}의 수치 데이터 형태가 올바르지 않습니다."
        assert img_data.shape == (config['N_STOCK'], 64, 117), f"테스트 데이터 {i}의 이미지 데이터 형태가 올바르지 않습니다."
        assert img_data.dtype == np.uint8, f"테스트 데이터 {i}의 이미지 데이터 타입이 올바르지 않습니다."
        assert np.min(img_data) >= 0 and np.max(img_data) <= 255, f"테스트 데이터 {i}의 이미지 데이터 값 범위가 올바르지 않습니다."

    print("모든 검증 통과")

if __name__ == "__main__":
    aligned_train, aligned_test, times_te, config = load_multimodal_data()
    print(f"Aligned train data: {len(aligned_train)}")
    print(f"Aligned test data: {len(aligned_test)}")
    print(f"Train numerical data shape: {aligned_train[0][0].shape}")
    print(f"Train image data shape: {aligned_train[0][1].shape}")
    
    test_multimodal_data()