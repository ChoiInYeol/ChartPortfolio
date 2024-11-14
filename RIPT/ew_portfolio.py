import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
plt.style.use('science')
import os
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def load_portfolio_data(file_path):
    """
    CSV 파일에서 포트폴리오 데이터를 로드합니다.
    """
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df

def calculate_cumulative_returns(df, resample_freq=None):
    """
    누적 수익률을 계산하고 선택적으로 리샘플링합니다.
    시작 지점을 1로 설정합니다.
    """
    cum_returns = (1 + df).cumprod()
    if resample_freq:
        cum_returns = cum_returns.resample(resample_freq).last()
    
    # 시작 지점을 1로 설정
    cum_returns = cum_returns / cum_returns.iloc[0]
    
    return cum_returns

def get_simplified_title(folder_name):
    """
    폴더 이름에서 간결한 제목을 추출합니다.
    """
    parts = folder_name.split('_')
    model_type = parts[1]  # CNN5, CNN20, CNN60, TS5, TS20 등
    days = parts[2].split('d')[0]  # 5, 20, or 60
    
    # TS가 포함된 경우 TS로 변경
    if "TS" in model_type:
        model_type = "TS"
    else:
        model_type = model_type[:3]  # CNN
    
    return f"{model_type} {days}-day"

def plot_portfolio_performance(data_dict, folder_name, resample_freq=None):
    """
    포트폴리오의 성과를 시각화합니다.
    """
    plt.figure(figsize=(8, 6), dpi=600)
    
    colors = LinearSegmentedColormap.from_list("custom", ["#FF4136", "#FFFFFF", "#0074D9"])
    
    for i, (name, df) in enumerate(data_dict.items()):
        if name == 'H-L':
            color = 'black'
        else:
            norm_i = i / (len(data_dict) - 2)
            color = colors(1 - norm_i)
        
        cum_returns = calculate_cumulative_returns(df, resample_freq)
        plt.plot(cum_returns.index, cum_returns.values, label=name, color=color, linewidth=1)
    
    title_suffix = f" (Resampled {resample_freq})" if resample_freq else ""
    simplified_title = get_simplified_title(folder_name)
    plt.title(f'Cumulative Returns of {simplified_title}{title_suffix}', fontsize=10)
    plt.xlabel('Date', fontsize=8)
    plt.ylabel('Cumulative Returns', fontsize=8)
    plt.legend(fontsize=6, ncol=2)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tick_params(axis='both', which='major', labelsize=6)
    
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    
    file_suffix = f"_resampled_{resample_freq}" if resample_freq else ""
    save_path = os.path.join('RIPT/WORK_DIR/portfolio', folder_name, f'cumulative_returns{file_suffix}.png')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()

def main():
    portfolio_dir = 'RIPT/WORK_DIR/portfolio'
    
    # 폴더 리스트 portfolio_dir에서 모든 폴더를 가져옴
    folders = os.listdir(portfolio_dir)

    for folder in folders:
        data_dict = {}
        folder_path = os.path.join(portfolio_dir, folder)
        pf_data_path = os.path.join(folder_path, 'pf_data', 'pf_data_ew.csv')
        
        if os.path.exists(pf_data_path):
            df = load_portfolio_data(pf_data_path)
            columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'H-L']
            labels = ['Low', '2', '3', '4', '5', '6', '7', '8', '9', 'High', 'H-L']
            data_dict = {label: df[col] for col, label in zip(columns, labels)}
            
            # 원본 데이터 플롯 (월별)
            plot_portfolio_performance(data_dict, folder)

if __name__ == '__main__':
    main()
