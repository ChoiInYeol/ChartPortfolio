import pandas as pd
import os

def filter_stocks(input_file='stock.csv', output_file='filtered_stock.csv', min_trading_days=6000):
    """
    stock.csv 파일을 읽고 거래일이 6000일 이상인 종목만 필터링하여 새로운 CSV 파일로 저장합니다.

    Args:
        input_file (str): 입력 CSV 파일 이름
        output_file (str): 출력 CSV 파일 이름
        min_trading_days (int): 최소 거래일 수

    Returns:
        None
    """
    # CSV 파일 읽기
    df = pd.read_csv(input_file)

    # 각 종목별 거래일 수 계산
    trading_days = df.groupby('PERMNO')['date'].count()

    # 거래일이 6000일 이상인 종목 필터링
    valid_permnos = trading_days[trading_days >= min_trading_days].index

    # 유효한 종목만 선택
    filtered_df = df[df['PERMNO'].isin(valid_permnos)]

    # 결과를 새로운 CSV 파일로 저장
    filtered_df.to_csv(output_file, index=False)

    print(f"원본 종목 수: {len(trading_days)}")
    print(f"필터링 후 종목 수: {len(valid_permnos)}")
    print(f"필터링된 데이터가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    filter_stocks()