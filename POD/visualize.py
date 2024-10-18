# visualize.py
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import numpy as np
import scienceplots
plt.style.use('science')

def visualize_backtest(performance, config):
    """
    백테스트 성과를 시각화합니다.

    Args:
        performance (pd.DataFrame): 포트폴리오 성과 데이터
        config (dict): 설정 정보

    Returns:
        None
    """
    plt.figure(figsize=(14, 7))
    for col in performance.columns:
        if col == 'EWP':
            plt.plot(performance.index, performance[col], label=col, color='black', linewidth=2)
        elif col == 'SPY':
            plt.plot(performance.index, performance[col], label=col, color='gray', linewidth=2)
        else:
            plt.plot(performance.index, performance[col], label=col, linestyle='--')
    plt.title("Portfolio Performance Comparison", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Portfolio Value", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    filename = os.path.join(config['RESULT_DIR'], f"performance.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_returns_distribution(performance, config):
    """
    수익률 분포를 시각화합니다.

    Args:
        performance (pd.DataFrame): 포트폴리오 성과 데이터
        config (dict): 설정 정보

    Returns:
        None
    """
    returns = performance.pct_change().dropna()
    plt.figure(figsize=(14, 7))
    for col in returns.columns:
        if col == 'EWP':
            sns.kdeplot(returns[col], label=col, color='black', linewidth=2)
        elif col == 'SPY':
            sns.kdeplot(returns[col], label=col, color='gray', linewidth=2)
        else:
            sns.kdeplot(returns[col], label=col, linestyle='--', fill=False)
    plt.title("Returns Distribution", fontsize=16)
    plt.xlabel("Returns", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    filename = os.path.join(config['RESULT_DIR'], f"returns_distribution.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_drawdown(performance, config):
    """
    포트폴리오 드로우다운을 시각화합니다.

    Args:
        performance (pd.DataFrame): 포트폴리오 성과 데이터
        config (dict): 설정 정보

    Returns:
        None
    """
    def calculate_drawdown(series):
        wealth_index = series
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        return drawdowns

    plt.figure(figsize=(14, 7))
    for col in performance.columns:
        drawdown = calculate_drawdown(performance[col])
        if col == 'EWP':
            plt.plot(drawdown.index, drawdown, label=col, color='black', linewidth=2)
        elif col == 'SPY':
            plt.plot(drawdown.index, drawdown, label=col, color='gray', linewidth=2)
        else:
            plt.plot(drawdown.index, drawdown, label=col, linestyle='--')
    plt.title("Portfolio Drawdown", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Drawdown", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    filename = os.path.join(config['RESULT_DIR'], f"drawdown.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_weights_over_time(weights_over_time, date_index, config):
    """
    시간에 따른 포트폴리오 가중치를 시각화합니다.

    Args:
        weights_over_time (dict): 시간에 따른 가중치 데이터
        date_index (pd.DatetimeIndex): 날짜 인덱스
        config (dict): 설정 정보

    Returns:
        None
    """
    n_stocks = list(weights_over_time.values())[0].shape[1]
    tickers = pd.read_csv("data/return_df.csv", index_col=0).columns[:n_stocks]

    # Truncate date_index if it's longer than weights
    min_length = min(len(date_index), list(weights_over_time.values())[0].shape[0])
    date_index = date_index[:min_length]

    for identifier, weights in weights_over_time.items():
        # Truncate weights to match date_index length
        weights = weights[:min_length]

        # Create the DataFrame
        weights_df = pd.DataFrame(weights, columns=tickers, index=date_index)

        # Calculate figure size based on number of stocks
        fig_width = max(12, n_stocks * 0.5)  # Increase width based on number of stocks
        fig_height = 8  # Fixed height

        # Create a new figure for each model
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Plot area chart
        weights_df.plot.area(stacked=True, ax=ax, alpha=0.5)
        
        plt.title(f"Weights Over Time - {identifier}", fontsize=14)
        plt.ylabel("Weight", fontsize=12)
        plt.xlabel("Date", fontsize=12)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=5)  # Adjust ncol as needed

        filename = os.path.join(config['RESULT_DIR'], f"weights_over_time_{identifier}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"생성된 그래프: {len(weights_over_time)}개")