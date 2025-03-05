# 차트 포트폴리오 (Chart Portfolio)

차트 이미지 기반 주가 예측 및 포트폴리오 구성 프로젝트

## 프로젝트 개요

이 프로젝트는 주식 차트 이미지를 CNN(Convolutional Neural Network) 모델로 분석하여 주가 상승 확률을 예측하고, 이를 기반으로 포트폴리오를 구성하는 시스템을 구현합니다.

## 주요 기능

- 주식 데이터 다운로드 및 전처리
- 차트 이미지 생성
- CNN 모델 학습 및 예측
- 포트폴리오 구성 및 성과 평가

## 프로젝트 구조

```
ChartPortfolio/
├── Data/                  # 공통 데이터 폴더 (자동 생성)
│
├── src/                   # 소스 코드
│   ├── data/              # 데이터 처리 모듈
│   │   ├── raw/           # 원시 데이터
│   │   ├── processed/     # 처리된 데이터
│   │   ├── meta/          # 메타데이터
│   │   ├── logs/          # 데이터 처리 로그
│   │   ├── data_download.py  # 데이터 다운로드
│   │   ├── data_pipeline.py  # 데이터 파이프라인
│   │   ├── data_ready.py     # 데이터 전처리
│   │   └── spy_download.py   # spy 데이터 다운로드
│   │
│   └── CNN_Models/            # 모델 관련 모듈
│       ├── Data/          # 차트 데이터 처리
│       ├── Experiments/   # 실험 관련 코드
│       ├── Misc/          # 유틸리티 및 설정
│       ├── Model/         # 모델 구현
│       ├── Portfolio/     # 포트폴리오 관리
│       ├── WORK_DIR/      # 작업 디렉토리 (자동 생성)
│       ├── cache/         # 캐시 디렉토리 (자동 생성)
│       ├── experiment.py  # 실험 실행 스크립트
│       └── generate_data.py  # 데이터 생성 스크립트
│
└── requirements.txt       # 의존성 패키지
```

## 설치 방법

1. 저장소 클론

```bash
git clone https://github.com/yourusername/ChartPortfolio.git
cd ChartPortfolio
```

2. 의존성 패키지 설치

```bash
pip install -r requirements.txt
```

3. 디렉토리 구조 설정

```bash
python src/models/setup_directories.py
```

## 사용 방법

### 1. 데이터 다운로드 및 처리

```bash
python -m src.data.data_pipeline --download --process
```

### 2. 차트 이미지 생성

```bash
python -m src.models.generate_data <GPU_IDS>
```

### 3. 모델 학습 및 포트폴리오 생성

```bash
python -m src.models.experiment <GPU_IDS>
```

## 모듈 설명

### src/data

데이터 다운로드 및 전처리를 담당하는 모듈입니다.

- `data_download.py`: 주식 데이터 다운로드
- `data_pipeline.py`: 데이터 처리 파이프라인
- `data_ready.py`: 데이터 전처리 및 필터링

### src/models

모델 학습 및 포트폴리오 구성을 담당하는 모듈입니다.

- `Data/`: 차트 이미지 생성 및 데이터셋 관리
- `Experiments/`: 모델 학습 및 평가
- `Model/`: CNN 모델 구현
- `Portfolio/`: 포트폴리오 구성 및 성과 평가

## 라이센스

MIT License

## 개발 가이드라인

1. 코드 스타일:
   - Black 포맷터 사용
   - Type hints 필수
   - Docstring 작성

2. 테스트:
   - 단위 테스트 작성
   - pytest 사용

3. 문서화:
   - 함수/클래스 문서화
   - 예제 코드 제공

## 기여 방법

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

