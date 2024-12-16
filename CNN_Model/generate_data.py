from Data.generate_chart import GenerateStockData
from Misc import config as cf
import torch
import os
import sys
import time
from tqdm import tqdm

def set_device(gpu_ids: str) -> torch.device:
    """
    GPU ID를 설정하고 디바이스를 반환합니다.

    Args:
        gpu_ids (str): 사용할 GPU ID 문자열 (예: "0,1,2,3")

    Returns:
        torch.device: 설정된 디바이스
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    return device

def generate_training_data(year_list, ws_list, freq="month", chart_type="bar", country="USA"):
    """
    CNN2D와 CNN1D 모델 훈련을 위한 데이터를 생성합니다.
    """
    total_start_time = time.time()
    
    for year in year_list:
        for ws in ws_list:
            print(f"\nGenerating data for {ws}D {freq} {chart_type} {year}")
            start_time = time.time()
            
            ma_lags = [60]  # 이동평균선 기간 설정
            vb = True  # 거래량 바 포함
            
            try:
                dgp_obj = GenerateStockData(
                    country,
                    year,
                    ws,
                    freq=freq,
                    chart_freq=1,  # 일봉 기준
                    ma_lags=ma_lags,
                    volume_bar=vb,
                    need_adjust_price=True,
                    allow_tqdm=True,
                    chart_type=chart_type,
                )
                
                # CPU 버전 사용
                dgp_obj.save_annual_data()
                
            except Exception as e:
                print(f"Error processing year {year}, window size {ws}: {str(e)}")
                continue
            
            end_time = time.time()
            print(f"Time taken for year {year}, window size {ws}: {(end_time - start_time) / 60:.2f} minutes")
    
    total_end_time = time.time()
    print(f"\nTotal time taken: {(total_end_time - total_start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    # 테스트를 위한 작은 데이터셋으로 시작
    test_years = [2005]  # 테스트용 연도
    test_ws = [20]      # 테스트용 윈도우 사이즈
    
    # print("\nGenerating test data...")
    # generate_training_data(
    #     year_list=test_years,
    #     ws_list=test_ws,
    #     freq='month',     # 일별 데이터 생성
    #     chart_type='bar'
    # ) 
    
    # # 테스트가 성공하면 전체 데이터 생성
    # print("\nGenerating full dataset...")
    # generate_training_data(
    #     year_list=cf.IS_YEARS + cf.OOS_YEARS,
    #     ws_list=[20],
    #     freq='month',
    #     chart_type='bar',
    # )
    
    # 테스트가 성공하면 전체 데이터 생성
    print("\nGenerating full dataset...")
    generate_training_data(
        year_list=cf.IS_YEARS + cf.OOS_YEARS,
        ws_list=[60],
        freq='month',
        chart_type='bar',
    )