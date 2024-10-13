from Experiments.cnn_experiment import train_us_model, train_my_model
from Data.generate_chart import GenerateStockData
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

def generate_training_data(year_list, ws_list, freq="month", chart_type="bar", country="USA"):
    """
    CNN2D와 CNN1D 모델 훈련을 위한 데이터를 생성합니다.

    Args:
        year_list (list): 데이터를 생성할 연도 리스트
        ws_list (list): 윈도우 사이즈 리스트
        freq (str): 데이터 주기 ('month', 'week', 'day' 등)
        chart_type (str): 차트 타입 ('bar', 'line' 등)
        country (str): 국가 코드

    이 함수는 지정된 연도, 윈도우 사이즈, 주기, 차트 타입, 국가에 대해
    CNN2D와 CNN1D 모델에 필요한 데이터를 생성하고 저장합니다.
    """
    for year in year_list:
        for ws in ws_list:
            print(f"Generating data for {ws}D {freq} {chart_type} {year}")
            
            ma_lags = [ws]  # 윈도우 사이즈와 동일한 이동평균 기간 사용
            vb = True  # 거래량 바 포함
            
            dgp_obj = GenerateStockData(
                country,
                year,
                ws,
                freq,
                chart_freq=1,  # 차트 주기 (I20/R20 to R5/R5의 경우, ws=20, chart_freq=4로 설정)
                ma_lags=ma_lags,
                volume_bar=vb,
                need_adjust_price=True,
                allow_tqdm=True,
                chart_type=chart_type,
            )
            
            # CNN2D 데이터 생성
            dgp_obj.save_annual_data()
            
            # CNN1D 데이터 생성
            dgp_obj.save_annual_ts_data()

if __name__ == "__main__":
    # 데이터 생성
    year_list = list(range(2001, 2024))
    ws_list = [5, 20, 60]
    # generate_training_data(year_list, ws_list)

    # CNN2D 모델 훈련
    train_my_model(
        ws_list=ws_list,
        pw_list=[20],
        drop_prob=0.50,
        ensem=5,
        total_worker=1,
        is_ensem_res=True,
        has_volume_bar=True,
        has_ma=True,
        chart_type="bar",
        calculate_portfolio=True,
        ts1d_model=False,
        lr=1e-5,
    )
    
    # CNN1D 모델 훈련
    # train_my_model(
    #     ws_list=ws_list,
    #     pw_list=[20],
    #     drop_prob=0.50,
    #     ensem=5,
    #     total_worker=1,
    #     is_ensem_res=True,
    #     has_volume_bar=True,
    #     has_ma=True,
    #     chart_type="bar",
    #     calculate_portfolio=True,
    #     ts1d_model=True,
    #     lr=1e-5,
    # )
    
    
    # train_us_model(
    #     [60],
    #     [20],
    #     total_worker=1,
    #     calculate_portfolio=True,
    #     ts1d_model=False,
    #     ts_scale="image_scale",
    #     regression_label=None,
    #     pf_delay_list=[0],
    #     lr=1e-4,
    # )
    
    # # CNN1D
    # train_us_model(
    #     [60],
    #     [20],
    #     total_worker=1,
    #     calculate_portfolio=True,
    #     ts1d_model=True,
    #     ts_scale="image_scale",
    #     regression_label=None,
    #     pf_delay_list=[0],
    #     lr=1e-4,
    # )
