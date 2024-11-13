import yaml
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, List

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    전처리된 수익률(일별)과 상승확률(월별) 데이터를 로드
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (수익률 데이터, 상승확률 데이터)
    """
    logging.info("데이터 로드 시작")
    
    # 전처리된 데이터 로드
    return_df = pd.read_csv("data/return_df.csv", index_col=0, parse_dates=True)
    up_prob_df = pd.read_csv("data/CNN_20D_20P_up_prob_df.csv", index_col=0, parse_dates=True)
    
    # 데이터 정합성 체크
    assert set(return_df.columns) == set(up_prob_df.columns), "수익률과 상승확률 데이터의 종목이 일치하지 않습니다."
    
    # 상승확률 데이터를 수익률 데이터의 인덱스로 확장
    expanded_prob_df = pd.DataFrame(0.5, 
                                  index=return_df.index, 
                                  columns=up_prob_df.columns)
    
    # 월별 상승확률 데이터 채우기
    for date in up_prob_df.index:
        if date in expanded_prob_df.index:
            expanded_prob_df.loc[date] = up_prob_df.loc[date]
    
    logging.info(f"데이터 로드 완료")
    logging.info(f"수익률 데이터(일별): {return_df.shape}")
    logging.info(f"원본 상승확률(월별): {up_prob_df.shape}")
    logging.info(f"확장된 상승확률(일별): {expanded_prob_df.shape}")
    
    return return_df, expanded_prob_df

def make_DL_dataset(data: pd.DataFrame, 
                   prob_data: pd.DataFrame,
                   train_len: int, 
                   pred_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
    """
    딥러닝 모델용 데이터셋 생성
    
    Args:
        data: 일별 수익률 데이터
        prob_data: 확장된 상승확률 데이터 (일별, 기본값 0.5)
        train_len: 학습 기간 (일)
        pred_len: 예측 기간 (일)
    """
    sequences = []
    labels = []
    probs = []
    dates = []
    
    total_len = train_len + pred_len
    
    for i in range(len(data) - total_len + 1):
        # 수익률 데이터 추출
        seq = data.iloc[i:i+train_len].values
        label = data.iloc[i+train_len:i+total_len].values
        date = data.iloc[i+train_len:i+total_len].index.tolist()
        
        # 상승확률 추출 (이미 0.5로 채워져 있음)
        prob = prob_data.iloc[i+train_len:i+total_len].values
        
        sequences.append(seq)
        labels.append(label)
        probs.append(prob)
        dates.append(date)
    
    return (np.array(sequences), 
            np.array(labels), 
            np.array(probs), 
            dates)

def split_data(return_df: pd.DataFrame, 
              up_prob_df: pd.DataFrame, 
              config: dict,
              train_len: int,
              pred_len: int) -> Tuple:
    """
    날짜 기준으로 데이터를 학습/검증/테스트 세트로 분할
    """
    # In-sample 기간 데이터 필터링
    in_sample_mask = (return_df.index >= config['TRAIN_START_DATE']) & \
                    (return_df.index <= config['TRAIN_END_DATE'])
    in_sample_data = return_df[in_sample_mask]
    in_sample_prob = up_prob_df[in_sample_mask]
    
    # In-sample 데이터를 학습/검증 세트로 분할
    train_end_idx = int(len(in_sample_data) * config['TRAIN_RATIO'])
    
    train_data = in_sample_data.iloc[:train_end_idx]
    train_prob = in_sample_prob.iloc[:train_end_idx]
    
    val_data = in_sample_data.iloc[train_end_idx:]
    val_prob = in_sample_prob.iloc[train_end_idx:]
    
    # Out-of-sample 테스트 데이터
    test_mask = (return_df.index >= config['TEST_START_DATE']) & \
                (return_df.index <= config['TEST_END_DATE'])
    test_data = return_df[test_mask]
    test_prob = up_prob_df[test_mask]
    
    # 데이터 크기 검증
    if len(train_data) < (train_len + pred_len) or \
       len(val_data) < (train_len + pred_len) or \
       len(test_data) < (train_len + pred_len):
        raise ValueError(
            f"데이터 길이가 부족합니다.\n"
            f"필요한 최소 길이: {train_len + pred_len}\n"
            f"실제 길이 - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
        )
    
    logging.info(f"데이터 분할 완료")
    logging.info(f"학습 데이터: {train_data.index[0]} ~ {train_data.index[-1]} ({len(train_data)}개)")
    logging.info(f"검증 데이터: {val_data.index[0]} ~ {val_data.index[-1]} ({len(val_data)}개)")
    logging.info(f"테스트 데이터: {test_data.index[0]} ~ {test_data.index[-1]} ({len(test_data)}개)")
    
    # 데이터셋 생성
    train_x, train_y, train_prob, train_dates = make_DL_dataset(
        train_data, train_prob, train_len, pred_len)
    val_x, val_y, val_prob, val_dates = make_DL_dataset(
        val_data, val_prob, train_len, pred_len)
    test_x, test_y, test_prob, test_dates = make_DL_dataset(
        test_data, test_prob, train_len, pred_len)
    
    # 데이터셋 크기 검증
    logging.info(f"최종 데이터셋 크기:")
    logging.info(f"Train - X: {train_x.shape}, Y: {train_y.shape}, Prob: {train_prob.shape}")
    logging.info(f"Val - X: {val_x.shape}, Y: {val_y.shape}, Prob: {val_prob.shape}")
    logging.info(f"Test - X: {test_x.shape}, Y: {test_y.shape}, Prob: {test_prob.shape}")
    
    if train_x.shape[0] == 0 or val_x.shape[0] == 0 or test_x.shape[0] == 0:
        raise ValueError(
            f"빈 데이터셋이 있습니다.\n"
            f"Train: {train_x.shape}, Val: {val_x.shape}, Test: {test_x.shape}"
        )
    
    return (train_x, train_y, train_prob, train_dates,
            val_x, val_y, val_prob, val_dates,
            test_x, test_y, test_prob, test_dates)

if __name__ == "__main__":
    # 설정 파일 로드
    with open("config/config.yaml", "r", encoding="utf8") as f:
        config = yaml.safe_load(f)
    
    # 로깅 설정
    logging.basicConfig(
        filename='make_dataset.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("데이터셋 생성 시작")
    
    # 데이터 로드
    return_df, up_prob_df = load_data()
    
    # 데이터 분할 및 데이터셋 생성
    train_x, train_y, train_prob, train_dates, \
    val_x, val_y, val_prob, val_dates, \
    test_x, test_y, test_prob, test_dates = split_data(
        return_df,
        up_prob_df,
        config=config,
        train_len=config['TRAIN_LEN'],
        pred_len=config['PRED_LEN']
    )
    
    # 데이터 저장
    with open("data/dataset.pkl", "wb") as f:
        pickle.dump({
            'train': (train_x, train_y, train_prob),
            'val': (val_x, val_y, val_prob),
            'test': (test_x, test_y, test_prob)
        }, f)
    
    with open("data/date.pkl", "wb") as f:
        pickle.dump({
            'train': train_dates,
            'val': val_dates,
            'test': test_dates
        }, f)
    
    logging.info("데이터셋 생성 완료")