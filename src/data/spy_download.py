import yfinance as yf
import os
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# 프로젝트 루트 경로 찾기
project_root = Path(__file__).resolve().parents[2]  # src/data에서 두 단계 위로 올라감

# 저장 경로 설정
PROCESSED_DIR = os.path.join(project_root, "src", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)
output_path = os.path.join(PROCESSED_DIR, "snp500_index.csv")

print(f"데이터 저장 경로: {PROCESSED_DIR}")

# 현재 날짜 가져오기 (종료일로 사용)
end_date = datetime.now().strftime("%Y-%m-%d")

print(f"^GSPC... (1999-12-21 ~ {end_date})")
# SPY 데이터 다운로드 (대안으로 "^GSPC" 사용 가능)
spy = yf.download("^GSPC", start="1999-12-21", end=end_date)

# 데이터 확인
print(f"다운로드된 데이터: {len(spy)}행, 기간: {spy.index.min()} ~ {spy.index.max()}")
print(f"컬럼: {spy.columns.tolist()}")

# 컬럼명 정리 (티커 제거)
spy.columns = [col[0] if isinstance(col, tuple) else col for col in spy.columns]
spy.reset_index(inplace=True)

# 데이터 저장
spy.to_csv(output_path, index=False)
print(f"S&P 500 인덱스 데이터가 저장되었습니다: {output_path}")

# 저장된 데이터 확인
print("\n저장된 데이터 샘플:")
saved_data = pd.read_csv(output_path, parse_dates=['Date'])
print(saved_data.head())
