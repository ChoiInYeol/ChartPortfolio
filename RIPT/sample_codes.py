from Experiments.cnn_experiment import train_us_model
from Data.generate_chart import GenerateStockData
from Portfolio.portfolio import PortfolioManager
import pandas as pd
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

def generate_and_plot_portfolio(model_name, freq, start_year, end_year, cut=10):
    # 포트폴리오 결과 로드
    portfolio_dir = f"results/US/{model_name}/portfolio"
    pm = PortfolioManager(pd.DataFrame(), freq, portfolio_dir, start_year, end_year)
    portfolio_ret = pm.load_portfolio_ret(cut=cut)

    # 포트폴리오 플롯 생성
    save_path = os.path.join(portfolio_dir, f"{model_name}_portfolio_plot.png")
    plot_title = f"{model_name} Portfolio Performance"
    pm.make_portfolio_plot(portfolio_ret, cut, "ew", save_path, plot_title)
    print(f"Portfolio plot saved to {save_path}")

if __name__ == "__main__":
    # Generate Image Data
    year_list = list(range(2001, 2024))
    chart_type = "bar"
    ws = 60 # window_size
    freq = "month"
    ma_lags = [ws] # ws (window_size)와 동일한 기간의 이동평균을 사용함으로써, 차트에 표시되는 모든 데이터 포인트에 대해 이동평균을 계산
    vb = True
    for year in year_list:
        print(f"{ws}D {freq} {chart_type} {year}")
        dgp_obj = GenerateStockData(
            "USA",
            year,
            ws,
            freq,
            chart_freq=1,  # for time-scale I20/R20 to R5/R5, set ws=20 and chart_freq=4
            ma_lags=ma_lags,
            volume_bar=vb,
            need_adjust_price=True,
            allow_tqdm=True,
            chart_type=chart_type,
        )
        # generate CNN2D Data
        #dgp_obj.save_annual_data()
        # generate CNN1D Data
        #dgp_obj.save_annual_ts_data()

    # Train CNN Models for US
    # CNN2D
    # train_us_model(
    #     [20], # window_size
    #     [5], # predict_window
    #     total_worker=1,
    #     calculate_portfolio=False,
    #     ts1d_model=False,
    #     ts_scale="image_scale",
    #     regression_label=None,
    #     pf_delay_list=[0],
    #     lr=1e-4,
    # )
    train_us_model(
        [60],
        [20],
        total_worker=1,
        calculate_portfolio=True,
        ts1d_model=False,
        ts_scale="image_scale",
        regression_label=None,
        pf_delay_list=[0],
        lr=1e-4,
    )
    
    # CNN2D 모델의 포트폴리오 플롯 생성
    # generate_and_plot_portfolio("CNN2D_60_20", "month", 2018, 2024)
    
    # train_us_model(
    #     [20],
    #     [60],
    #     total_worker=1,
    #     calculate_portfolio=False,
    #     ts1d_model=False,
    #     ts_scale="image_scale",
    #     regression_label=None,
    #     pf_delay_list=[0],
    #     lr=1e-4,
    # )
    # CNN1D
    train_us_model(
        [60],
        [20],
        total_worker=1,
        calculate_portfolio=True,
        ts1d_model=True,
        ts_scale="image_scale",
        regression_label=None,
        pf_delay_list=[0],
        lr=1e-4,
    )
    
    # CNN1D 모델의 포트폴리오 플롯 생성
    # generate_and_plot_portfolio("CNN1D_60_20", "month", 2019, 2024)
    
    # # Timescale
    # train_us_model(
    #     [20],
    #     [20],
    #     total_worker=1,
    #     calculate_portfolio=False,
    #     ts1d_model=False,
    #     ts_scale="image_scale",
    #     regression_label=None,
    #     pf_delay_list=[0],
    #     lr=1e-5,
    # )
    # train_us_model(
    #     [60],
    #     [60],
    #     total_worker=1,
    #     calculate_portfolio=False,
    #     ts1d_model=False,
    #     ts_scale="image_scale",
    #     regression_label=None,
    #     pf_delay_list=[0],
    #     lr=1e-5,
    # )
