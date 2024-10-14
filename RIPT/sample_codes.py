from Experiments.cnn_experiment import train_us_model, train_my_model
from Data.generate_chart import GenerateStockData
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

0

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
