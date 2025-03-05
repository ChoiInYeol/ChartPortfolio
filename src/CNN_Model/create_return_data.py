import os
import pandas as pd
import numpy as np
from Data import equity_data

# 처리된 미국 주식 데이터 가져오기
print("처리된 미국 주식 데이터를 가져옵니다...")
df = equity_data.processed_US_data()

# 필요한 컬럼만 선택
print("필요한 컬럼만 선택합니다...")
ret_df = df.reset_index()[['Date', 'StockID', 'Ret']].copy()

# 날짜 형식 변환
ret_df['Date'] = pd.to_datetime(ret_df['Date'])

# 피벗 테이블 생성 (date를 인덱스로, StockID를 컬럼으로)
print("피벗 테이블을 생성합니다...")
pivot_df = ret_df.pivot(index='Date', columns='StockID', values='Ret')

# 결과 저장
print("결과를 저장합니다...")
os.makedirs('processed', exist_ok=True)
pivot_df.reset_index().rename(columns={'Date': 'date'}).to_csv('processed/return_df.csv', index=False)

print("수익률 데이터 생성이 완료되었습니다.") 