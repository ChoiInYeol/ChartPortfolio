import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
plt.style.use('science')
import os

def load_portfolio_data(file_path):
    """
    CSV 파일에서 포트폴리오 데이터를 로드합니다.
    """
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df

def calculate_cumulative_returns(df):
    """
    누적 수익률을 계산합니다.
    """
    return (1 + df).cumprod() - 1

def plot_portfolio_performance(data_dict, folder_name):
    """
    포트폴리오의 성과를 시각화합니다.
    """
    plt.figure(figsize=(15, 10))
    
    colors = ['#FF4136', '#FF851B', '#FFDC00', '#2ECC40', '#0074D9', 
              '#001f3f', '#39CCCC', '#01FF70', '#85144b', '#F012BE']
    
    for i, (name, df) in enumerate(data_dict.items()):
        if name == 'H-L':
            color = 'black'
        elif i < 5:
            color = colors[i]
        else:
            color = colors[-(i-4)]
        
        cum_returns = calculate_cumulative_returns(df)
        plt.plot(cum_returns.index, cum_returns.values, label=name, color=color)
    
    plt.title(f'Cumulative Returns of {folder_name}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    save_path = os.path.join('RIPT_WORK_SPACE/new_model_res/portfolio', folder_name, 'cumulative_returns.png')
    plt.savefig(save_path)
    plt.close()

def main():
    portfolio_dir = 'RIPT_WORK_SPACE/new_model_res/portfolio'
    folders = [
        'USA_CNN5_5d20p_e5_2009-2023',
        'USA_CNN20_20d20p_e5_2009-2023',
        'USA_CNN60_60d20p_e5_2009-2023',
        'USA_TS5_5d20p_e5_2009-2023',
        'USA_TS20_20d20p_e5_2009-2023',
        'USA_TS60_60d20p_e5_2009-2023'
    ]

    for folder in folders:
        data_dict = {}
        folder_path = os.path.join(portfolio_dir, folder)
        pf_data_path = os.path.join(folder_path, 'pf_data', 'pf_data_ew.csv')
        
        if os.path.exists(pf_data_path):
            df = load_portfolio_data(pf_data_path)
            columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'H-L']
            labels = ['High', '9', '8', '7', '6', '5', '4', '3', '2', 'Low', 'H-L']
            data_dict = {label: df[col] for col, label in zip(columns, labels)}
            plot_portfolio_performance(data_dict, folder)

if __name__ == '__main__':
    main()