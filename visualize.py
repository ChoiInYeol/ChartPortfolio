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
    plt.figure(figsize=(14, 7))
    for col in performance.columns:
        plt.plot(performance.index, performance[col], label=col)
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
    returns = performance.pct_change().dropna()
    plt.figure(figsize=(14, 7))
    for col in returns.columns:
        sns.kdeplot(returns[col], label=col, fill=True)
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
    def calculate_drawdown(series):
        wealth_index = series
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks
        return drawdowns
    plt.figure(figsize=(14, 7))
    for col in performance.columns:
        drawdown = calculate_drawdown(performance[col])
        plt.plot(drawdown.index, drawdown, label=col)
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
        weights_df.plot.area(stacked=True, ax=ax)
        
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

