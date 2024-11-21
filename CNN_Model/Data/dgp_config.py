import os
import os.path as op
from pathlib import Path

def get_dir(path):
    """
    Creates a directory if it does not exist.

    Args:
        path (str): The directory path to create.

    Returns:
        str: The directory path.
    """
    if not op.exists(path):
        os.makedirs(path)
    return path

# 최상위 작업 디렉토리 설정
WORK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data 폴더는 최상위에서 참조
DATA_DIR = op.join(WORK_DIR, "Data")  # 공통 데이터 폴더

# CNN_Model 관련 경로 설정
CNN_DIR = op.join(WORK_DIR, "CNN_Model")
WORK_DIR = op.join(CNN_DIR, "WORK_DIR")  # CNN_Model 작업 디렉토리
CACHE_DIR = op.join(CNN_DIR, "cache")  # CNN_Model 직속 cache 디렉토리

# Dataset 구조 설정
STOCKS_SAVEPATH = op.join(WORK_DIR, "Dataset")
SAMPLE_IMAGES_DIR = op.join(STOCKS_SAVEPATH, "sample_images")
PROCESSED_DATA_DIR = op.join(STOCKS_SAVEPATH, "processed_data")
STOCKS_USA_DIR = op.join(STOCKS_SAVEPATH, "stocks_USA")
STOCKS_USA_TS_DIR = op.join(STOCKS_SAVEPATH, "stocks_USA_ts")

# WORK_DIR 내부 디렉토리 구조
PORTFOLIO_DIR = op.join(WORK_DIR, "portfolio")
LOG_DIR = op.join(WORK_DIR, "log")
LATEX_DIR = op.join(WORK_DIR, "latex")

# 원본 데이터는 최상위 Data 폴더 참조
RAW_DATA_DIR = DATA_DIR

# 필요한 디렉토리 생성
for dir_path in [
    DATA_DIR,  # 최상위 Data 폴더
    CACHE_DIR,  # CNN_Model/cache
    STOCKS_SAVEPATH,  # CNN_Model/WORK_DIR/Dataset
    SAMPLE_IMAGES_DIR,  # Dataset/sample_images
    PROCESSED_DATA_DIR,  # Dataset/processed_data
    STOCKS_USA_DIR,  # Dataset/stocks_USA
    STOCKS_USA_TS_DIR,  # Dataset/stocks_USA_ts
    PORTFOLIO_DIR,  # WORK_DIR/portfolio
    LOG_DIR,  # WORK_DIR/log
    LATEX_DIR,  # WORK_DIR/latex
]:
    get_dir(dir_path)

# 나머지 설정들은 그대로 유지
BAR_WIDTH = 3
LINE_WIDTH = 1
IMAGE_WIDTH = {5: BAR_WIDTH * 5, 20: BAR_WIDTH * 20, 60: BAR_WIDTH * 60}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}
VOLUME_CHART_GAP = 1
BACKGROUND_COLOR = 0
CHART_COLOR = 255

FREQ_DICT = {5: "week",
             20: "month",
             60: "quarter",
             65: "quarter",
             260: "year"}

INTERNATIONAL_COUNTRIES = [
    "Japan",
    "UnitedKingdom",
    "China",
    "SouthKorea",
    "India",
    "Canada",
    "Germany",
    "Australia",
    "HongKong",
    "France",
    "Singapore",
    "Italy",
    "Sweden",
    "Switzerland",
    "Netherlands",
    "Norway",
    "Spain",
    "Belgium",
    "Greece",
    "Denmark",
    "Russia",
    "Finland",
    "NewZealand",
    "Austria",
    "Portugal",
    "Ireland",
]