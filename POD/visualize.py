# visualize.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scienceplots
plt.style.use('science')
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def visualize_backtest(performance: pd.DataFrame, config: dict) -> None:
    """
    백테스트 성과를 시각화합니다.

    Args:
        performance (pd.DataFrame): 포트폴리오 성과 데이터
        config (dict): 설정 정보
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
    logger.info(f"Performance plot saved to {filename}")

def visualize_returns_distribution(performance: pd.DataFrame, config: dict) -> None:
    """
    수익률 분포를 시각화합니다.
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
    logger.info(f"Returns distribution plot saved to {filename}")

def calculate_drawdown(series: pd.Series) -> pd.Series:
    """드로다운을 계산합니다."""
    wealth_index = series
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return drawdowns

def visualize_drawdown(performance: pd.DataFrame, config: dict) -> None:
    """
    포트폴리오 드로다운을 시각화합니다.
    """
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
    logger.info(f"Drawdown plot saved to {filename}")

def visualize_weights_over_time(weights_over_time: dict, date_index: pd.DatetimeIndex, config: dict) -> None:
    """
    시간에 따른 포트폴리오 가중치를 시각화합니다.
    """
    if not weights_over_time:
        logger.warning("No weights data available for visualization")
        return

    n_stocks = config['N_STOCK']
    try:
        tickers = pd.read_csv("data/return_df.csv", index_col=0).columns[:n_stocks]
    except Exception as e:
        logger.error(f"Error loading tickers: {e}")
        tickers = [f"Stock_{i}" for i in range(n_stocks)]

    # 날짜 인덱스와 가중치 길이 맞추기
    min_length = min(len(date_index), list(weights_over_time.values())[0].shape[0])
    date_index = date_index[:min_length]

    for identifier, weights in weights_over_time.items():
        try:
            weights = weights[:min_length]
            weights_df = pd.DataFrame(weights, columns=tickers, index=date_index)

            fig_width = max(12, n_stocks * 0.5)
            fig, ax = plt.subplots(figsize=(fig_width, 8))

            weights_df.plot.area(stacked=True, ax=ax, alpha=0.6)
            plt.title(f"Portfolio Weights Over Time - {identifier}", fontsize=14)
            plt.ylabel("Weight", fontsize=12)
            plt.xlabel("Date", fontsize=12)
            plt.ylim(0, 1)
            plt.grid(True, linestyle='--', alpha=0.7)

            # 범례 조정
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                           box.width, box.height * 0.9])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                     fancybox=True, shadow=True, ncol=5)

            filename = os.path.join(config['RESULT_DIR'], f"weights_over_time_{identifier}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Weights plot saved for {identifier}")

        except Exception as e:
            logger.error(f"Error visualizing weights for {identifier}: {e}")
            continue

    logger.info(f"Generated {len(weights_over_time)} weight plots")