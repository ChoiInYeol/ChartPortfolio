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
    n_models = len(weights_over_time)
    n_stocks = list(weights_over_time.values())[0].shape[1]
    tickers = pd.read_csv("data/return_df.csv", index_col=0).columns[:n_stocks]

    # Truncate date_index if it's longer than weights
    min_length = min(len(date_index), list(weights_over_time.values())[0].shape[0])
    date_index = date_index[:min_length]

    # Create a color map with enough distinct colors
    color_map = plt.get_cmap('tab20')  # You can also use 'tab20b', 'tab20c', or other colormaps
    colors = [color_map(i % 20) for i in range(n_stocks)]  # Handle more than 20 by repeating

    # Create subplots
    fig, axes = plt.subplots(nrows=n_models, ncols=1, figsize=(14, 4 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]  # Ensure axes is iterable

    for ax, (identifier, weights) in zip(axes, weights_over_time.items()):
        # Truncate weights to match date_index length
        weights = weights[:min_length]

        # Create the DataFrame and plot with the custom color palette
        weights_df = pd.DataFrame(weights, columns=tickers, index=date_index)
        weights_df.plot(ax=ax, legend=False, color=colors)
        ax.set_title(f"Weights Over Time - {identifier}", fontsize=14)
        ax.set_ylabel("Weight", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

    # Add a single legend for all tickers at the bottom
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5, fontsize=10, frameon=False
    )

    plt.xlabel("Date", fontsize=12)
    plt.tight_layout()
    filename = os.path.join(config['RESULT_DIR'], f"weights_over_time.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

