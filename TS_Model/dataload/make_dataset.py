import yaml
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, List, Dict, Any

logger = logging.getLogger(__name__)

def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    원본 수익률과 확률 데이터를 로드합니다.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (수익률 데이터, 상승확률 데이터)
    """
    try:
        # 데이터 경로 설정
        data_dir = Path("/home/indi/codespace/ImagePortOpt/Data")
        
        # 수익률과 확률 데이터 로드
        returns_df = pd.read_csv(
            data_dir / "processed/return_df.csv",
            index_col=0, parse_dates=True
        )
        probs_df = pd.read_csv(
            data_dir / "processed/ensem_res.csv",
            index_col=0, parse_dates=True
        )
        
        # NaN 값 처리
        returns_df = returns_df.fillna(0)  # 수익률의 NaN을 0으로 채움
        
        # NaN 처리 결과 로깅
        logging.info(f"데이터 로드 완료")
        logging.info(f"수익률 데이터 shape: {returns_df.shape}")
        logging.info(f"상승확률 데이터 shape: {probs_df.shape}")
        logging.info(f"수익률 데이터 NaN 비율: {(returns_df.isna().sum().sum() / returns_df.size):.4%}")
        logging.info(f"상승확률 데이터 NaN 비율: {(probs_df.isna().sum().sum() / probs_df.size):.4%}")
        
        return returns_df, probs_df
        
    except Exception as e:
        logging.error(f"데이터 로드 중 오류 발생: {str(e)}")
        raise

def filter_sp500_stocks(returns_df: pd.DataFrame, probs_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    SP500 종목만 필터링합니다.
    
    Args:
        returns_df: 전체 수익률 데이터
        probs_df: 전체 상승확률 데이터
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: SP500 종목으로 필터링된 (수익률, 상승확률) 데이터
    """
    try:
        # SP500 종목 리스트 로드
        sp500_symbols = pd.read_csv(
            "/home/indi/codespace/ImagePortOpt/Data/raw_data/sp500_20180101.csv"
        )['Symbol'].tolist()
        
        # SP500 종목 중 데이터가 있는 종목만 필터링
        available_symbols = [sym for sym in sp500_symbols if sym in returns_df and sym in probs_df]
        filtered_returns = returns_df[available_symbols]
        filtered_probs = probs_df[available_symbols]
        
        logging.info(f"SP500 필터링 완료")
        logging.info(f"필터링된 종목 수: {len(sp500_symbols)}")
        logging.info(f"필터링된 수익률 데이터 shape: {filtered_returns.shape}")
        logging.info(f"필터링된 상승확률 데이터 shape: {filtered_probs.shape}")
        
        # csv로 저장
        filtered_returns.to_csv(
            "/home/indi/codespace/ImagePortOpt/TS_Model/data/filtered_returns.csv"
        )
        filtered_probs.to_csv(
            "/home/indi/codespace/ImagePortOpt/TS_Model/data/filtered_probs.csv"
        )
        
        return filtered_returns, filtered_probs
        
    except Exception as e:
        logging.error(f"SP500 필터링 중 오류 발생: {str(e)}")
        raise

def normalize_probabilities(probs: np.ndarray, method: str = 'standardize') -> np.ndarray:
    """
    확률값을 정규화하여 hidden layer concat 시 적절한 스케일을 가지도록 조정합니다.
    
    Args:
        probs: 원본 확률값 [0, 1] 범위
        method: 정규화 방법 ('standardize' 또는 'minmax')
    
    Returns:
        정규화된 확률값
    """
    if method == 'standardize':
        # 평균 0, 표준편차 1로 정규화
        mean = np.mean(probs)
        std = np.std(probs)
        return (probs - mean) / (std + 1e-8)
    
    elif method == 'minmax':
        # [-1, 1] 범위로 정규화
        min_val = np.min(probs)
        max_val = np.max(probs)
        return 2 * (probs - min_val) / (max_val - min_val + 1e-8) - 1
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def make_dl_dataset(
    returns_df: pd.DataFrame,
    probs_df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
    """
    딥러닝 모델용 데이터셋을 생성합니다.
    
    Args:
        returns_df: 일별 수익률 데이터
        probs_df: 월별 상승확률 데이터 (리밸런싱 날짜 기준)
        config: 설정 정보
    
    Returns:
        sequences: 입력 시퀀스 [batch, window_size, n_stocks]
        labels: 레이블 [batch, pred_window, n_stocks]
        probs: 상승확률 [batch, pred_window, n_stocks]
        dates: 예측 기간 날짜 리스트
    """
    sequences = []
    labels = []
    probs = []
    dates = []
    
    window_size = config['DATA']['WINDOW_SIZE']
    pred_window = config['DATA']['PRED_WINDOW']
    
    # 상승확률 데이터의 인덱스(리밸런싱 날짜들)
    rebalancing_dates = probs_df.index
    
    # 중립 확률값 생성 (0.5로 채워진 배열)
    neutral_probs = np.full_like(probs_df.iloc[0].values, 0.5)
    
    for i in range(len(returns_df) - window_size - pred_window + 1):
        # 현재 시퀀스의 날짜들
        sequence_dates = returns_df.index[i:i+window_size]
        pred_dates = returns_df.index[i+window_size:i+window_size+pred_window]
        
        # 수익률 시퀀스와 레이블 추출
        sequence = returns_df.iloc[i:i+window_size].values
        label = returns_df.iloc[i+window_size:i+window_size+pred_window].values
        
        # NaN 검사 및 로깅
        if np.isnan(sequence).any() or np.isnan(label).any():
            logging.warning(f"NaN 발견: sequence_dates={sequence_dates[0]} ~ {sequence_dates[-1]}")
            # NaN을 0으로 대체
            sequence = np.nan_to_num(sequence, 0)
            label = np.nan_to_num(label, 0)
        
        # 예측 기간의 첫 날짜가 리밸런싱 날짜인지 확인
        pred_start_date = pred_dates[0]
        if pred_start_date in rebalancing_dates:
            prob = probs_df.loc[pred_start_date].values
            
            # 정규화 적용
            if config['DATA']['PROB_NORMALIZE'] == 'standardize':
                prob = normalize_probabilities(prob, 'standardize')
            elif config['DATA']['PROB_NORMALIZE'] == 'minmax':
                prob = normalize_probabilities(prob, 'minmax')
        else:
            prob = neutral_probs
        
        sequences.append(sequence)
        labels.append(label)
        probs.append(prob)
        dates.append(pred_dates.tolist())
    
    # 최종 데이터셋의 NaN 검사
    final_sequences = np.array(sequences)
    final_labels = np.array(labels)
    final_probs = np.array(probs)
    
    if np.isnan(final_sequences).any() or np.isnan(final_labels).any() or np.isnan(final_probs).any():
        logging.warning("최종 데이터셋에 NaN 존재")
        logging.warning(f"sequences NaN 비율: {np.isnan(final_sequences).mean():.4%}")
        logging.warning(f"labels NaN 비율: {np.isnan(final_labels).mean():.4%}")
        logging.warning(f"probs NaN 비율: {np.isnan(final_probs).mean():.4%}")
    
    return final_sequences, final_labels, final_probs, dates

def split_data(
    returns_df: pd.DataFrame,
    probs_df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple:
    """
    데이터를 학습/검증/테스트 세트로 분할합니다.
    
    Args:
        returns_df: 수익률 데이터 (일별)
        probs_df: 상승확률 데이터 (월별)
        config: 설정 정보
    
    Returns:
        Tuple: (train_x, train_y, train_prob, train_dates,
               val_x, val_y, val_prob, val_dates,
               test_x, test_y, test_prob, test_dates)
    """
    # 일별 데이터에 맞춰 확률 데이터 리샘플링
    prob_daily = pd.DataFrame(index=returns_df.index, columns=probs_df.columns)
    
    # 각 월별 확률을 해당 월의 모든 거래일에 복제
    for date in probs_df.index:
        current_probs = probs_df.loc[date]
        month_start = pd.Timestamp(date).to_period('M').start_time
        month_end = pd.Timestamp(date).to_period('M').end_time
        mask = (returns_df.index >= month_start) & (returns_df.index <= month_end)
        
        # 각 열(종목)에 대해 동일한 확률값 할당
        for col in prob_daily.columns:
            if col in current_probs.index:
                prob_daily.loc[mask, col] = current_probs[col]
    
    # 빈 값을 0.5로 채우기
    prob_daily.fillna(0.5, inplace=True)

    # In-sample 기간 데이터 필터링
    in_sample_mask = (returns_df.index >= config['DATA']['TRAIN']['START_DATE']) & \
                    (returns_df.index <= config['DATA']['TRAIN']['END_DATE'])
    in_sample_data = returns_df[in_sample_mask]
    in_sample_prob = prob_daily[in_sample_mask]
    
    # In-sample 데이터를 학습/검증 세트로 분할
    train_end_idx = int(len(in_sample_data) * config['DATA']['TRAIN']['RATIO'])
    
    train_data = in_sample_data.iloc[:train_end_idx]
    train_prob = in_sample_prob.iloc[:train_end_idx]
    
    val_data = in_sample_data.iloc[train_end_idx:]
    val_prob = in_sample_prob.iloc[train_end_idx:]
    
    # Out-of-sample 테스트 데이터
    test_mask = (returns_df.index >= config['DATA']['TEST']['START_DATE']) & \
                (returns_df.index <= config['DATA']['TEST']['END_DATE'])
    test_data = returns_df[test_mask]
    test_prob = prob_daily[test_mask]
    
    # 데이터셋 생성
    train_x, train_y, train_prob, train_dates = make_dl_dataset(
        train_data, train_prob, config)
    val_x, val_y, val_prob, val_dates = make_dl_dataset(
        val_data, val_prob, config)
    test_x, test_y, test_prob, test_dates = make_dl_dataset(
        test_data, test_prob, config)
    
    return (train_x, train_y, train_prob, train_dates,
            val_x, val_y, val_prob, val_dates,
            test_x, test_y, test_prob, test_dates)

def prepare_dataset(config: Dict[str, Any]) -> None:
    """
    전체 데이터셋 준비 과정을 실행합니다.
    
    Args:
        config: 설정 정보
    """
    try:
        # 1. 원본 데이터 로드
        logging.info("1. 원본 데이터 로드 시작")
        returns_df, probs_df = load_raw_data()
        
        # 2. SP500 종목 필터링
        logging.info("2. SP500 종목 필터링 시작")
        filtered_returns, filtered_probs = filter_sp500_stocks(returns_df, probs_df)
        
        # 3. 데이터셋 분할 및 생성
        logging.info("3. 데이터셋 분할 및 생성 시작")
        train_x, train_y, train_prob, train_dates, \
        val_x, val_y, val_prob, val_dates, \
        test_x, test_y, test_prob, test_dates = split_data(
            filtered_returns, filtered_probs, config
        )
        
        # 4. 데이터셋 저장
        logging.info("4. 데이터셋 저장 시작")
        dataset = {
            'train': (train_x, train_y, train_prob),
            'val': (val_x, val_y, val_prob),
            'test': (test_x, test_y, test_prob)
        }
        dates = {
            'train': train_dates,
            'val': val_dates,
            'test': test_dates
        }
        
        save_dir = Path("/home/indi/codespace/ImagePortOpt/TS_Model/data")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(save_dir / "dataset.pkl", "wb") as f:
            pickle.dump(dataset, f)
        with open(save_dir / "dates.pkl", "wb") as f:
            pickle.dump(dates, f)
            
        logging.info("데이터셋 준비 완료")
        
    except Exception as e:
        logging.error(f"데이터셋 준비 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 설정 파일 로드
    config_path = Path("/home/indi/codespace/ImagePortOpt/TS_Model/config/base_config.yaml")
    with open(config_path, "r", encoding="utf8") as f:
        config = yaml.safe_load(f)
    
    # 데이터셋 준비
    prepare_dataset(config)