import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import pickle

# 설정 파일 로드
with open('POD/config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# 필요한 설정값 추출
TRAIN_START_DATE = config['TRAIN_START_DATE']
TRAIN_END_DATE = config['TRAIN_END_DATE']
VALIDATION_RATIO = config['VALIDATION_RATIO']
TEST_START_DATE = config['TEST_START_DATE']
TEST_END_DATE = config['TEST_END_DATE']
M = config['M']

# CSV 파일 로드 및 병합
csv_dir = 'RIPT/WORK_DIR/CNN20/20D20P/ensem_res'
csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

all_data = []
for csv_file in csv_files:
    file_path = os.path.join(csv_dir, csv_file)
    df = pd.read_csv(file_path)
    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)

# filtered_stock.csv 로드
filtered_stock = pd.read_csv('RIPT/Dataset/raw_data/filtered_stock.csv')
valid_stock_ids = filtered_stock['PERMNO'].unique()

# 유효하지 않은 StockID 제거
combined_df = combined_df[combined_df['StockID'].isin(valid_stock_ids)]

# 날짜를 datetime으로 변환
combined_df['ending_date'] = pd.to_datetime(combined_df['ending_date'])

# 데이터 분할
train_data = combined_df[(combined_df['ending_date'] >= TRAIN_START_DATE) & (combined_df['ending_date'] <= TRAIN_END_DATE)]
test_data = combined_df[(combined_df['ending_date'] > TEST_START_DATE) & (combined_df['ending_date'] <= TEST_END_DATE)]

# 훈련 데이터를 train과 validation으로 분할
train_dates = train_data['ending_date'].unique()
train_dates = np.sort(train_dates)  # 날짜 정렬
validation_split = int(len(train_dates) * (1 - VALIDATION_RATIO))
validation_start_date = train_dates[validation_split]

train_data_final = train_data[train_data['ending_date'] < validation_start_date]
val_data = train_data[train_data['ending_date'] >= validation_start_date]

def select_top_stocks(group, m):
    top_stocks = group.nlargest(m, 'up_prob')['StockID'].values
    # 항상 m 크기의 배열을 반환하도록 패딩
    return np.pad(top_stocks, (0, m - len(top_stocks)), 'constant', constant_values=-1)

def expand_monthly_to_daily(monthly_data, daily_dates):
    monthly_dates = pd.date_range(start=daily_dates.min(), end=daily_dates.max(), freq='ME')
    daily_indices = pd.cut(daily_dates, bins=monthly_dates, labels=False, include_lowest=True)
    return np.array([monthly_data[i] for i in daily_indices])

# 각 날짜별로 상위 M개 선택 (월별)
train_selected_monthly = [select_top_stocks(group, M) for _, group in train_data_final.groupby(pd.Grouper(key='ending_date', freq='ME'))]
val_selected_monthly = [select_top_stocks(group, M) for _, group in val_data.groupby(pd.Grouper(key='ending_date', freq='ME'))]
test_selected_monthly = [select_top_stocks(group, M) for _, group in test_data.groupby(pd.Grouper(key='ending_date', freq='ME'))]

# 월별 데이터를 일별 데이터로 확장
train_selected = expand_monthly_to_daily(train_selected_monthly, train_data_final['ending_date'])
val_selected = expand_monthly_to_daily(val_selected_monthly, val_data['ending_date'])
test_selected = expand_monthly_to_daily(test_selected_monthly, test_data['ending_date'])

# 결과를 딕셔너리로 저장
selected_indices = {
    'train': train_selected,
    'val': val_selected,
    'test': test_selected
}

# 결과 저장
with open('selected_indices.pkl', 'wb') as f:
    pickle.dump(selected_indices, f)

print("Train selected shape:", train_selected.shape)
print("Validation selected shape:", val_selected.shape)
print("Test selected shape:", test_selected.shape)
print("Number of unique dates in train data:", len(train_data_final['ending_date'].unique()))
print("Number of unique dates in validation data:", len(val_data['ending_date'].unique()))
print("Number of unique dates in test data:", len(test_data['ending_date'].unique()))
print("Train data date range:", train_data['ending_date'].min(), "to", train_data['ending_date'].max())
print("Test data date range:", test_data['ending_date'].min(), "to", test_data['ending_date'].max())
