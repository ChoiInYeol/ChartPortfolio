# visualize.py
import os
import matplotlib.pyplot as plt
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
    filename = os.path.join(config['RESULT_DIR'], f"performance_{config['MODEL']}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_weights(performance, weights, config):
    weights = np.array(weights)
    ticker = pd.read_csv("data/return_df.csv", index_col=0).columns
    n = config['N_FEAT']
    plt.figure(figsize=(15, 10))
    for i in range(n):
        plt.plot(weights[:, i], label=ticker[i])
    plt.title("Portfolio Weights Over Time", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Weight", fontsize=14)
    plt.xticks(
        np.arange(0, len(list(performance.index[1:]))),
        list(performance.index[1:]),
        rotation=45,
        ha='right'
    )
    plt.legend(fontsize=10, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    filename = os.path.join(config['RESULT_DIR'], f"weights_{config['MODEL']}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_returns_distribution(performance, config):
    returns = performance.pct_change().dropna()
    plt.figure(figsize=(14, 7))
    for col in returns.columns:
        sns.kdeplot(returns[col], label=col)
    plt.title("Returns Distribution", fontsize=16)
    plt.xlabel("Returns", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    filename = os.path.join(config['RESULT_DIR'], f"returns_distribution_{config['MODEL']}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_drawdown(performance, config):
    def calculate_drawdown(series):
        wealth_index = (1 + series.pct_change()).cumprod()
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
    filename = os.path.join(config['RESULT_DIR'], f"drawdown_{config['MODEL']}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_rolling_sharpe(performance, config, window=252):
    returns = performance.pct_change().dropna()
    rolling_sharpe = returns.rolling(window=window).apply(lambda x: np.sqrt(252) * x.mean() / x.std())
    plt.figure(figsize=(14, 7))
    for col in rolling_sharpe.columns:
        plt.plot(rolling_sharpe.index, rolling_sharpe[col], label=col)
    plt.title(f"Rolling Sharpe Ratio (Window: {window} days)", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Sharpe Ratio", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    filename = os.path.join(config['RESULT_DIR'], f"rolling_sharpe_{config['MODEL']}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_binary_predictions(binary_preds_df, config):
    plt.figure(figsize=(14, 7))
    sns.heatmap(binary_preds_df.T, cmap="YlOrRd", cbar_kws={'label': 'Prediction Probability'})
    plt.title("Binary Predictions Heatmap", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Stock", fontsize=14)
    plt.tight_layout()
    filename = os.path.join(config['RESULT_DIR'], f"binary_predictions_{config['MODEL']}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
