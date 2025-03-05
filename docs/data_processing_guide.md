# 데이터 다운로드 및 전처리 가이드

이 문서는 주식 데이터의 다운로드부터 전처리까지의 전체 과정을 설명합니다.

## 목차
1. [환경 설정](#환경-설정)
2. [데이터 파이프라인 실행](#데이터-파이프라인-실행)
3. [데이터 처리 단계](#데이터-처리-단계)
4. [생성되는 피처](#생성되는-피처)
5. [디렉토리 구조](#디렉토리-구조)
6. [문제 해결](#문제-해결)

## 환경 설정

### 필수 패키지 설치
```bash
pip install pandas numpy yfinance FinanceDataReader scipy torch tqdm
```

### 프로젝트 설정
1. 프로젝트 루트 디렉토리에서 다음 명령을 실행하여 패키지를 설치합니다:
```bash
pip install -e .
```

2. 또는 PYTHONPATH에 프로젝트 루트 디렉토리를 추가합니다:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/ChartPortfolio
```

### 디렉토리 구조 생성
파이프라인은 다음과 같은 디렉토리 구조를 자동으로 생성합니다:
```
data/
├── raw/              # 원본 데이터
├── processed/        # 처리된 데이터
└── market_caps/      # 시가총액 데이터
```

## 데이터 파이프라인 실행

### 기본 사용법
```python
# 방법 1: 모듈로 실행
python -m src.data.data_pipeline

# 방법 2: Python 코드에서 실행
from src.data import DataPipeline

# 파이프라인 초기화
pipeline = DataPipeline()

# 파이프라인 실행
pipeline.run_pipeline()
```

### 사용자 정의 설정
```python
from pathlib import Path
from src.data import DataPipeline

# 사용자 정의 설정으로 파이프라인 실행
pipeline = DataPipeline(
    base_dir=Path('./my_project'),
    min_history_days=3000
)

pipeline.run_pipeline(
    target_date='2019-01-01',  # 데이터 시작일
    window_days=20             # 시가총액 계산 윈도우
)
```

## 데이터 처리 단계

### 1. 주식 데이터 다운로드
- NASDAQ, NYSE 상장 종목 정보 수집
- 개별 종목 가격 데이터 다운로드
- 최소 거래일수 기준으로 필터링

### 2. S&P 500 종목 확인
- S&P 500 구성종목 데이터 확인
- 누락된 종목 추가 다운로드
- 오류 기록 관리

### 3. 시가총액 데이터 수집
- 개별 종목의 시가총액 정보 수집
- 특정 시점 기준 시가총액 데이터 추출
- 시가총액 순위 정보 생성

### 4. 데이터 전처리
- 결측치 처리
  - 수치형 데이터: 전진 채우기(ffill) 또는 평균값
  - 범주형 데이터: 최빈값
- 이상치 처리
  - Z-score 방식
  - IQR 방식
- 피처 엔지니어링
- 데이터 정규화

## 생성되는 피처

### 수익률 관련 피처
- `Daily_Return`: 일간 수익률
- `Return_Volatility`: 수익률 변동성 (20일)

### 거래량 관련 피처
- `Volume_MA`: 거래량 이동평균 (20일)
- `Volume_Ratio`: 현재 거래량/이동평균 비율

### 가격 관련 피처
- `MA_{window}`: 이동평균 (5, 10, 20, 60일)
- `RSI_{window}`: 상대강도지수 (5, 10, 20, 60일)

### 모멘텀 피처
- `Momentum_1M`: 1개월 모멘텀
- `Momentum_3M`: 3개월 모멘텀
- `Momentum_6M`: 6개월 모멘텀

### 변동성 피처
- `High_Low_Ratio`: 고가/저가 비율
- `Price_Range`: 가격 범위

## 디렉토리 구조

```
project/
├── data/
│   ├── raw/              # 원본 데이터
│   ├── processed/        # 처리된 데이터
│   └── market_caps/      # 시가총액 데이터
├── src/
│   └── data/
│       ├── data_pipeline.py    # 메인 파이프라인
│       ├── data_processor.py   # 데이터 처리
│       ├── data_download.py    # 데이터 다운로드
│       └── marketcap.py        # 시가총액 처리
└── docs/
    └── data_processing_guide.md  # 본 문서
```

## 문제 해결

### 일반적인 문제

1. 데이터 다운로드 실패
```python
# 재시도 옵션으로 실행
pipeline.run_pipeline(
    config={'download': {'max_retries': 3}}
)
```

2. 메모리 부족
```python
# 배치 크기 조정
pipeline.run_pipeline(
    config={'processing': {'batch_size': 1000}}
)
```

### 로그 확인
- 모든 처리 과정은 `pipeline.log`에 기록됨
- 에러 및 경고 메시지 확인 가능

```bash
tail -f pipeline.log  # 실시간 로그 확인
```

### 데이터 검증
```python
# 처리된 데이터 확인
import pandas as pd

data = pd.read_csv('data/processed/processed_data.csv')
print("데이터 형태:", data.shape)
print("\n결측치 현황:")
print(data.isnull().sum())
```

### 주의사항
1. 충분한 디스크 공간 확보 (최소 10GB 권장)
2. 안정적인 인터넷 연결 필요
3. API 호출 제한 고려
4. 대용량 데이터 처리 시 메모리 관리 필요 