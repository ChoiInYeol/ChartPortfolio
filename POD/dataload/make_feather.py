import pandas as pd
import pyarrow.feather as feather

def csv_to_feather(csv_path, feather_path):
    """
    CSV 파일을 Feather 형식으로 변환합니다.

    Args:
        csv_path (str): 입력 CSV 파일 경로
        feather_path (str): 출력 Feather 파일 경로

    Returns:
        None
    """
    # CSV 파일 읽기
    df = pd.read_csv(csv_path, parse_dates=['date'])

    # Feather 파일로 저장
    feather.write_feather(df, feather_path)

    print(f"CSV 파일 '{csv_path}'을(를) Feather 파일 '{feather_path}'(으)로 변환했습니다.")

if __name__ == "__main__":
    csv_path = "data/filtered_stock.csv"
    feather_path = "data/filtered_stock.feather"
    csv_to_feather(csv_path, feather_path)