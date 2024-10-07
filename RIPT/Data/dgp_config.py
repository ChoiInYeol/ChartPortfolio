import os
import os.path as op
from pathlib import Path

from Misc.config import WORK_DIR

def get_dir(path):
    if not op.exists(path):
        os.makedirs(path)
    return path

WORK_DIR = os.getcwd()
DATA_DIR = op.join(WORK_DIR, "data")
PROCESSED_DATA_DIR = op.join(WORK_DIR, "processed_data")
STOCKS_SAVEPATH = op.join(WORK_DIR, "stocks_dataset")
RAW_DATA_DIR = op.join(STOCKS_SAVEPATH, "raw_data")
CACHE_DIR = op.join(WORK_DIR, "cache")

for dir_path in [PROCESSED_DATA_DIR, STOCKS_SAVEPATH, RAW_DATA_DIR, CACHE_DIR]:
    get_dir(dir_path)

BAR_WIDTH = 3
LINE_WIDTH = 1
IMAGE_WIDTH = {5: BAR_WIDTH * 5, 20: BAR_WIDTH * 20, 60: BAR_WIDTH * 60}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}
VOLUME_CHART_GAP = 1
BACKGROUND_COLOR = 0
CHART_COLOR = 255

FREQ_DICT = {5: "week", 20: "month", 60: "quarter", 65: "quarter", 260: "year"}

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
