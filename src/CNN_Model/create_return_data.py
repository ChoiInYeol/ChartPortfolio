import os
import pandas as pd
import numpy as np
from Data import equity_data
from Data.dgp_config import RAW_DATA_DIR

# 처리된 미국 주식 데이터 가져오기
print("처리된 미국 주식 데이터를 가져옵니다...")
df = equity_data.processed_US_data()

# 필요한 컬럼만 선택
print("필요한 컬럼만 선택합니다...")
ret_df = df.reset_index()[['Date', 'StockID', 'Ret']].copy()

# 날짜 형식 변환
ret_df['Date'] = pd.to_datetime(ret_df['Date'])

# StockID를 문자열로 변환
print("StockID를 문자열로 변환합니다...")
ret_df['StockID'] = ret_df['StockID'].astype(str)

# symbol_permno.csv 로드 및 전처리
print("symbol_permno.csv 파일을 로드합니다...")
symbol_permno = pd.read_csv(os.path.join(RAW_DATA_DIR, 'symbol_permno.csv'))
symbol_permno['PERMNO'] = symbol_permno['PERMNO'].astype(str)  # PERMNO를 문자열로 변환
symbol_permno.rename(columns={'PERMNO': 'StockID'}, inplace=True)  # PERMNO 컬럼을 StockID로 변경

# 매핑 수행
print("StockID를 Symbol로 매핑합니다...")
ret_df = ret_df.merge(symbol_permno[['StockID', 'Symbol']], on='StockID', how='left')

# Symbol이 없는 행 제거
ret_df = ret_df.dropna(subset=['Symbol'])

# 중복 확인
duplicates = ret_df.duplicated(subset=['Date', 'Symbol'], keep=False)
if duplicates.any():
    print(f"중복된 항목이 {duplicates.sum()}개 발견되었습니다. 중복 항목을 처리합니다...")
    # 중복 항목의 예시 출력
    print("중복 항목 예시:")
    print(ret_df[duplicates].head())

# 피벗 테이블 생성 (date를 인덱스로, Symbol을 컬럼으로)
# 중복된 항목이 있을 경우 평균값을 사용
print("피벗 테이블을 생성합니다...")
pivot_df = ret_df.pivot_table(index='Date', columns='Symbol', values='Ret', aggfunc='mean')

# 결과 저장
print("결과를 저장합니다...")
os.makedirs('processed', exist_ok=True)
pivot_df.reset_index().rename(columns={'Date': 'date'}).to_csv('processed/return_df.csv', index=False)

print("수익률 데이터 생성이 완료되었습니다.") 