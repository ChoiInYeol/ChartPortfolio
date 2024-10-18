import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

import logging
import torch
from torch.optim import Adam
from tqdm import tqdm
import scienceplots
plt.style.use(['science'])

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TRAIN = '2017-12-31'
MODELS = ['CNN']
WINDOW_SIZES = [20]
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

def to_tensor(data):
    if isinstance(data, pd.DataFrame):
        return torch.tensor(data.values, dtype=torch.float32, device=device)
    elif isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=torch.float32, device=device)
    return data

def to_numpy(tensor):
    return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()

def handle_outliers(returns, prices, lookback=5, tolerance=0.05, volatility_threshold=2.0):
    """
    주가와 수익률 데이터를 사용해 이상치를 처리하고, 변동성이 극심한 주식은 제거합니다.
    
    Args:
        returns (pd.DataFrame): 일별 수익률 데이터.
        prices (pd.DataFrame): 일별 주가 데이터.
        lookback (int): n일 동안 가격 변화를 추적합니다.
        tolerance (float): 가격 변화 허용 오차율 (예: 0.05 = 5%).
        volatility_threshold (float): 제거할 주식의 변동성 임계값 (예: 200% 이상 변동).
    
    Returns:
        tuple: (이상치가 처리된 수익률 데이터, 제거된 주식 목록)
    """
    high_threshold = 0.75  # 75% 이상의 수익률   
    low_threshold = -0.75  # -75% 이하의 수익률

    # 1. 극단적인 수익률을 가진 주식 탐지
    extreme_returns = returns[(returns > high_threshold) | (returns < low_threshold)]

    # 2. 변동성이 높은 주식 식별 (최대 변동률이 volatility_threshold 이상인 주식)
    max_daily_volatility = prices.pct_change().abs().max()
    stocks_to_remove = max_daily_volatility[max_daily_volatility > volatility_threshold].index.tolist()

    # 주식 제거 결과 출력
    logging.info(f"총 {len(stocks_to_remove)}개의 주식이 변동성 기준을 초과하여 제거됩니다.")
    logging.info(f"제거된 주식 목록: {stocks_to_remove}")

    # 3. 변동성이 높은 주식 제거
    returns_cleaned = returns.drop(columns=stocks_to_remove, errors='ignore')
    prices_cleaned = prices.drop(columns=stocks_to_remove, errors='ignore')

    # 4. 극단치 처리 (조정할 날짜-주식 목록 저장)
    to_adjust = []
    for date in extreme_returns.index:
        for stock_id in extreme_returns.columns[extreme_returns.loc[date].notnull()]:
            if stock_id in stocks_to_remove:
                continue  # 제거된 주식은 건너뜀

            # 현재 날짜의 가격과 lookback 기간 가격 추적
            current_price = prices_cleaned.at[date, stock_id]
            start_idx = max(0, prices_cleaned.index.get_loc(date) - lookback)
            end_idx = min(len(prices_cleaned) - 1, prices_cleaned.index.get_loc(date) + lookback)

            price_window = prices_cleaned.iloc[start_idx:end_idx + 1][stock_id].dropna()
            if len(price_window) < 2:
                continue  # 데이터 부족 시 건너뜀

            # 첫날과 마지막 날의 가격 차이 비율 계산
            first_price = price_window.iloc[0]
            last_price = price_window.iloc[-1]
            price_change = abs(last_price - first_price) / first_price

            # 가격 변동이 tolerance 이내이면 이상치로 간주
            if price_change <= tolerance:
                to_adjust.append((date, stock_id))

    # 5. 수익률 조정 및 통계 출력
    adjusted_count = len(to_adjust)
    adjusted_stocks = len(set(stock_id for _, stock_id in to_adjust))

    for date, stock_id in to_adjust:
        returns_cleaned.at[date, stock_id] = 0

    # 6. 기타 이상치에 대한 보간 처리
    returns_cleaned = returns_cleaned.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)

    # 결과 출력
    logging.info(f"총 {adjusted_count}개의 이상치 수익률을 조정했습니다.")
    logging.info(f"총 {adjusted_stocks}개의 주식에 대해 수익률 조정을 수행했습니다.")

    return returns_cleaned, stocks_to_remove

def load_and_preprocess_data():
    file_path = os.path.join(BASE_FOLDER, 'processed_data', 'us_ret.feather')
    us_ret = pd.read_feather(file_path)
    us_ret = us_ret[us_ret['Date'] >= TRAIN]
    us_ret = us_ret.pivot(index='Date', columns='StockID', values='Ret')
    us_ret.index = pd.to_datetime(us_ret.index)

    if us_ret.abs().mean().mean() > 1:
        logging.info("Converting returns from percentages to decimals.")
        us_ret = us_ret / 100

    stock_prices_path = os.path.join(BASE_FOLDER, 'processed_data', 'us_ret.feather')
    stock_prices = pd.read_feather(stock_prices_path)
    stock_prices = stock_prices.pivot(index='Date', columns='StockID', values='Close')
    stock_prices.index = pd.to_datetime(stock_prices.index)
    stock_prices = stock_prices.fillna(method='ffill')

    us_ret_cleaned, removed_stocks = handle_outliers(us_ret, stock_prices)

    return us_ret_cleaned, removed_stocks

def load_benchmark_data():
    file_path = os.path.join(BASE_FOLDER, 'processed_data', 'snp500_index.csv')
    snp500 = pd.read_csv(file_path, parse_dates=['Date'])
    snp500.set_index('Date', inplace=True)
    snp500.sort_index(inplace=True)
    snp500 = snp500[snp500.index >= TRAIN]
    snp500['Returns'] = snp500['Adj Close'].pct_change()
    snp500['Cumulative Returns'] = (1 + snp500['Returns']).cumprod()
    return snp500

def load_and_process_ensemble_results(file_path, removed_stocks):
    """
    앙상블 결과를 로드하고 처리합니다. 이상치로 제거된 주식은 처음부터 제외합니다.

    Args:
        file_path (str): 앙상블 결과 파일 경로
        removed_stocks (list): 이상치로 제거된 주식 목록

    Returns:
        pd.DataFrame: 처리된 앙상블 결과 데이터
    """
    ensemble_results = pd.read_feather(file_path)
    
    if isinstance(ensemble_results.index, pd.MultiIndex):
        ensemble_results = ensemble_results.reset_index()
    
    ensemble_results['investment_date'] = pd.to_datetime(ensemble_results['investment_date'])
    ensemble_results['ending_date'] = pd.to_datetime(ensemble_results['ending_date'])
    ensemble_results['StockID'] = ensemble_results['StockID'].astype(str)
    
    # 이상치로 제거된 주식 제외
    ensemble_results = ensemble_results[~ensemble_results['StockID'].isin(removed_stocks)]
    
    return ensemble_results

class PortfolioOptimizer(torch.nn.Module):
    def __init__(self, num_assets):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.ones(num_assets) / num_assets)

    def forward(self):
        return torch.nn.functional.softmax(self.weights, dim=0)

def optimize_portfolio(returns, method='max_sharpe', epochs=1000, lr=0.01):
    print(f"Optimizing portfolio using {method} method")
    returns_tensor = to_tensor(returns)
    num_assets = returns_tensor.shape[1]
    
    model = PortfolioOptimizer(num_assets).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    
    for _ in tqdm(range(epochs), desc="Optimizing"):
        optimizer.zero_grad()
        weights = model()
        
        if method == 'max_sharpe':
            portfolio_return = (returns_tensor.mean(dim=0) * weights).sum()
            portfolio_volatility = torch.sqrt(torch.dot(weights, torch.matmul(returns_tensor.T, returns_tensor).matmul(weights)))
            sharpe_ratio = portfolio_return / portfolio_volatility
            loss = -sharpe_ratio
        elif method == 'min_variance':
            portfolio_volatility = torch.sqrt(torch.dot(weights, torch.matmul(returns_tensor.T, returns_tensor).matmul(weights)))
            loss = portfolio_volatility
        elif method == 'min_cvar':
            portfolio_returns = torch.matmul(returns_tensor, weights)
            var = torch.quantile(portfolio_returns, 0.05)
            cvar = portfolio_returns[portfolio_returns <= var].mean()
            loss = -cvar
        else:
            raise ValueError("Invalid optimization method")
        
        loss.backward()
        optimizer.step()
    
    optimized_weights = to_numpy(model().detach())
    print(f"Optimization completed - Weights shape: {optimized_weights.shape}")
    return optimized_weights

def save_optimization_results(optimized_returns, portfolio_weights, model, window_size, optimization_method):
    folder_name = os.path.join(BASE_FOLDER, 'WORK_DIR', f'{model}{window_size}')
    os.makedirs(folder_name, exist_ok=True)
    file_name = f'optimization_results_{optimization_method}.pkl'
    file_path = os.path.join(folder_name, file_name)
    
    with open(file_path, 'wb') as f:
        pickle.dump((optimized_returns, portfolio_weights), f)
    
    print(f"Optimization results saved to {file_path}")

def load_optimization_results(model, window_size, optimization_method):
    folder_name = os.path.join(BASE_FOLDER, 'WORK_DIR', f'{model}{window_size}')
    file_name = f'optimization_results_{optimization_method}.pkl'
    file_path = os.path.join(folder_name, file_name)
    
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            optimized_returns, portfolio_weights = pickle.load(f)
        print(f"Optimization results loaded from {file_path}")
        return optimized_returns, portfolio_weights
    else:
        print(f"No saved optimization results found for {optimization_method}")
        return None, None

def process_portfolio(selected_stocks, us_ret, optimization_method, model, window_size):
    # 먼저 저장된 결과가 있는지 확인
    optimized_returns, portfolio_weights = load_optimization_results(model, window_size, optimization_method)
    
    if optimized_returns is not None and portfolio_weights is not None:
        return optimized_returns, portfolio_weights

    print(f"Processing portfolio using {optimization_method} method")
    optimized_returns = []
    portfolio_weights = {}
    
    selected_stocks = selected_stocks[selected_stocks['investment_date'] >= TRAIN]
    rebalance_dates = selected_stocks['investment_date'].unique()
    
    for i, current_date in enumerate(tqdm(rebalance_dates, desc="Processing dates")):
        # 현재 날짜가 us_ret 인덱스에 없는 경우, 다음으로 가장 가까운 거래일을 찾습니다.
        while current_date not in us_ret.index and current_date <= us_ret.index[-1]:
            current_date += pd.Timedelta(days=1)
        
        if current_date > us_ret.index[-1]:
            logging.warning(f"No valid date found after {current_date}. Skipping.")
            continue
        
        current_index = us_ret.index.get_loc(current_date)
        
        # 60개의 과거 데이터를 선택합니다 (거래일 기준)
        start_index = max(0, current_index - 60)
        historical_returns = us_ret.iloc[start_index:current_index]
        
        # 선택된 주식만 필터링합니다
        selected_stocks_for_date = selected_stocks[selected_stocks['investment_date'] == rebalance_dates[i]]['StockID']
        historical_returns = historical_returns[selected_stocks_for_date]
        
        if historical_returns.empty or historical_returns.isnull().all().all():
            logging.warning(f"Empty or all-null historical returns for date {current_date}. Skipping.")
            continue

        try:
            print(f"Historical returns shape for {current_date}: {historical_returns.shape}")
            weights = optimize_portfolio(historical_returns, method=optimization_method)
            
            # 포트폴리오 비중 저장 (모든 종목에 대해 저장)
            portfolio_weights[current_date] = dict(zip(us_ret.columns, np.zeros(len(us_ret.columns))))
            for stock, weight in zip(historical_returns.columns, weights):
                portfolio_weights[current_date][stock] = weight
            
            if i < len(rebalance_dates) - 1:
                next_rebalance_date = rebalance_dates[i + 1]
                while next_rebalance_date not in us_ret.index and next_rebalance_date <= us_ret.index[-1]:
                    next_rebalance_date += pd.Timedelta(days=1)
            else:
                next_rebalance_date = us_ret.index[-1]
            
            next_period_returns = us_ret.loc[current_date:next_rebalance_date, historical_returns.columns]
            optimized_return = (next_period_returns * weights).sum(axis=1)
            optimized_returns.extend(list(zip(next_period_returns.index, optimized_return)))
        except Exception as e:
            logging.error(f"Error in portfolio optimization for date {current_date}: {str(e)}")
    
    optimized_returns = pd.Series(dict(optimized_returns))
    
    # 결과 저장
    save_optimization_results(optimized_returns, portfolio_weights, model, window_size, optimization_method)
    
    return optimized_returns, portfolio_weights

def plot_optimized_portfolios(portfolio_ret, optimized_portfolios, model, window_size, result_dir, rebalance_dates):
    plt.figure(figsize=(8, 6), dpi=400)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(portfolio_ret.columns) + len(optimized_portfolios)))
    
    for i, column in enumerate(portfolio_ret.columns):
        plt.plot(portfolio_ret.index, portfolio_ret[column], label=column, color=colors[i])
        
        # 리밸런싱 날짜에 점 추가 (Naive와 SPY 제외)
        if column in rebalance_dates:
            for date in rebalance_dates[column]:
                if date in portfolio_ret.index:
                    plt.plot(date, portfolio_ret.loc[date, column], marker='o', color=colors[i], markersize=3)
    
    for i, (method, returns) in enumerate(optimized_portfolios.items(), start=len(portfolio_ret.columns)):
        cumulative_returns = (1 + returns).cumprod()
        plt.plot(cumulative_returns.index, cumulative_returns.values, label=f'Optimized ({method})', color=colors[i], linestyle='--')
        
        # 리밸런싱 날짜에 점 추가
        for date in rebalance_dates['Top 100']:  # 최적화된 포트폴리오는 Top 100과 동일한 리밸런싱 날짜 사용
            if date in cumulative_returns.index:
                plt.plot(date, cumulative_returns.loc[date], marker='o', color=colors[i], markersize=3)
    
    plt.title(f'Portfolio Performance Comparison - {model} {window_size}-day', fontsize=12)
    plt.xlabel('Date', fontsize=10)
    plt.ylabel('Cumulative Returns', fontsize=10)
    plt.legend(fontsize=8, ncol=2, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(result_dir, f'portfolio_comparison_{model}{window_size}.png')
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.close()

def calculate_performance_metrics(combined_returns, portfolio_weights, selected_stocks, n_stocks_10, us_ret):
    def calculate_turnover(weights_dict):
        dates = sorted(weights_dict.keys())
        turnover = 0
        for i in range(1, len(dates)):
            prev_weights = pd.Series(weights_dict[dates[i-1]])
            curr_weights = pd.Series(weights_dict[dates[i]])
            
            # 모든 종목을 포함하도록 인덱스 통합
            all_stocks = prev_weights.index.union(curr_weights.index)
            prev_weights = prev_weights.reindex(all_stocks, fill_value=0)
            curr_weights = curr_weights.reindex(all_stocks, fill_value=0)
            
            # 턴오버 계산 (단방향)
            turnover += np.abs(curr_weights - prev_weights).sum()
        
        return turnover / len(dates)

    def calculate_stock_turnover(stocks_dict):
        dates = sorted(stocks_dict.keys())
        turnover = 0
        for i in range(1, len(dates)):
            prev_stocks = set(stocks_dict[dates[i-1]])
            curr_stocks = set(stocks_dict[dates[i]])
            
            # 진입한 종목 수 + 퇴출된 종목 수
            changes = len(curr_stocks - prev_stocks) + len(prev_stocks - curr_stocks)
            
            # 전체 종목 수 대비 변화 비율
            turnover += changes / len(prev_stocks.union(curr_stocks))
        
        return turnover / (len(dates) - 1)

    def calculate_max_drawdown(returns):
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()

    metrics = {}
    column_order = ['Naive', 'SPY', 'Top 100', f'Top {n_stocks_10}', 'Bottom 100', 'Optimized_max_sharpe', 'Optimized_min_variance', 'Optimized_min_cvar']
    column_names = {
        'Naive': 'Naive',
        'SPY': 'SPY',
        'Top 100': 'Top 100',
        f'Top {n_stocks_10}': f'Top {n_stocks_10}',
        'Bottom 100': 'Bottom 100',
        'Optimized_max_sharpe': 'Max sharpe',
        'Optimized_min_variance': 'Min Variance',
        'Optimized_min_cvar': 'Min CVaR'
    }

    for column in column_order:
        returns = combined_returns[column].pct_change().dropna()
        metrics[column_names[column]] = {
            'Return': returns.mean() * 252,
            'Std': returns.std() * np.sqrt(252),
            'SR': (returns.mean() / returns.std()) * np.sqrt(252),
            'Max Drawdown': calculate_max_drawdown(returns)
        }

        if column in ['Top 100', f'Top {n_stocks_10}', 'Bottom 100']:
            metrics[column_names[column]]['Turnover'] = calculate_stock_turnover(selected_stocks[column])
        elif column.startswith('Optimized_'):
            metrics[column_names[column]]['Turnover'] = calculate_turnover(portfolio_weights[column.split('_')[1]])
        else:
            metrics[column_names[column]]['Turnover'] = np.nan

    return pd.DataFrame(metrics).T[['Return', 'Std', 'SR', 'Max Drawdown', 'Turnover']]

def save_performance_metrics(metrics, model, window_size, folder_name):
    # CSV로 저장
    csv_path = os.path.join(folder_name, f'performance_metrics_{model}{window_size}.csv')
    metrics.to_csv(csv_path, float_format='%.3f')
    
    # LaTeX로 저장
    latex_path = os.path.join(folder_name, f'performance_metrics_{model}{window_size}.tex')
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[]\n\\begin{tabular}{ccccc}\n\\hline\n")
        f.write("             & Return   & Std      & SR       & Turnover \\\\ \\hline\n")
        for index, row in metrics.iterrows():
            f.write(f"{index:<12} & {row['Return']:.3f} & {row['Std']:.3f} & {row['SR']:.3f} & {row['Turnover']:.3f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}")
    
    print(f"Performance metrics saved to {csv_path} and {latex_path}")

    # 추가로 텍스트 파일로도 저장
    txt_path = os.path.join(folder_name, f'performance_metrics_{model}{window_size}.txt')
    with open(txt_path, 'w') as f:
        f.write("\tReturn\tStd\tSR\tTurnover\n")
        for index, row in metrics.iterrows():
            f.write(f"{index}\t{row['Return']:.3f}\t{row['Std']:.3f}\t{row['SR']:.3f}\t{row['Turnover']:.3f}\n")
    
    print(f"Performance metrics also saved to {txt_path}")

def get_next_valid_date(date, date_index):
    future_dates = date_index[date_index > pd.Timestamp(date)]
    return future_dates.min() if not future_dates.empty else None

def get_previous_valid_date(date, date_index):
    past_dates = date_index[date_index < pd.Timestamp(date)]
    return past_dates.max() if not past_dates.empty else None

def calculate_portfolio_returns(us_ret, selected_stocks, valid_stock_ids):
    portfolio_returns = pd.Series(dtype=float)
    rebalance_dates = []
    
    dates = sorted(selected_stocks.keys())
    for i, investment_date in enumerate(dates):
        stock_ids = selected_stocks[investment_date]
        valid_stock_ids_for_period = list(set(stock_ids) & valid_stock_ids)
        
        start_date = get_next_valid_date(investment_date, us_ret.index)
        if start_date is None:
            continue
        
        rebalance_dates.append(start_date)  # 리밸런싱 날짜 추가
        
        if i < len(dates) - 1:
            end_date = get_previous_valid_date(dates[i+1], us_ret.index)
            if end_date is None or end_date <= start_date:
                continue
        else:
            end_date = us_ret.index[-1]
        
        period_returns = us_ret.loc[start_date:end_date, valid_stock_ids_for_period]
        if period_returns.empty:
            continue
        
        weights = np.array([1 / len(valid_stock_ids_for_period)] * len(valid_stock_ids_for_period))
        daily_portfolio_returns = period_returns.dot(weights)
        
        portfolio_returns = portfolio_returns.add(daily_portfolio_returns, fill_value=0)
    
    portfolio_returns = portfolio_returns.sort_index()
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    return cumulative_returns, rebalance_dates

def process_model_window(model, window_size, us_ret, benchmark_data, removed_stocks):
    """
    주어진 모델과 윈도우 크기에 대해 포트폴리오를 처리하고 최적화합니다.

    Args:
        model (str): 사용할 모델의 이름
        window_size (int): 윈도우 크기
        us_ret (pd.DataFrame): 미국 주식 수익률 데이터
        benchmark_data (pd.DataFrame): 벤치마크 데이터
        removed_stocks (list): 제거된 주식 목록

    Returns:
        str: 처리 완료 메시지 또는 None (처리 실패 시)
    """
    file_path = os.path.join(BASE_FOLDER, 'WORK_DIR', f'ensemble_{model}{window_size}_res.feather')
    if not os.path.exists(file_path):
        logging.warning(f"File not found: {file_path}")
        return None

    logging.info(f'Processing model: {model}, window size: {window_size}')

    folder_name = os.path.join(BASE_FOLDER, 'WORK_DIR', f'{model}{window_size}')
    os.makedirs(folder_name, exist_ok=True)

    ensemble_results = load_and_process_ensemble_results(file_path, removed_stocks)

    n_stocks_100 = 100  # 고정된 주식 수
    n_stocks_10 = round(us_ret.shape[1] / 10)  # TOP N/10 주식 수
    
    def select_stocks(data, n_stocks, top=True):
        return data.sort_values(['investment_date', 'up_prob'], ascending=[True, not top]).groupby('investment_date').head(n_stocks)

    selected_stocks_top_100 = select_stocks(ensemble_results, n_stocks_100, top=True)
    selected_stocks_bottom_100 = select_stocks(ensemble_results, n_stocks_100, top=False)
    selected_stocks_top_10 = select_stocks(ensemble_results, n_stocks_10, top=True)

    logging.info(f"Total unique stocks in ensemble results: {len(ensemble_results['StockID'].unique())}")
    logging.info(f"Top 100 stocks: {len(selected_stocks_top_100['StockID'].unique())}")
    logging.info(f"Bottom 100 stocks: {len(selected_stocks_bottom_100['StockID'].unique())}")
    logging.info(f"Top {n_stocks_10} stocks: {len(selected_stocks_top_10['StockID'].unique())}")

    if selected_stocks_top_100.empty or selected_stocks_bottom_100.empty or selected_stocks_top_10.empty:
        logging.warning(f"No valid stocks for {model} with window size {window_size}. Skipping.")
        return None

    selected_stocks = {
        'Top 100': {date: selected_stocks_top_100[selected_stocks_top_100['investment_date'] == date]['StockID'].tolist() for date in selected_stocks_top_100['investment_date'].unique()},
        f'Top {n_stocks_10}': {date: selected_stocks_top_10[selected_stocks_top_10['investment_date'] == date]['StockID'].tolist() for date in selected_stocks_top_10['investment_date'].unique()},
        'Bottom 100': {date: selected_stocks_bottom_100[selected_stocks_bottom_100['investment_date'] == date]['StockID'].tolist() for date in selected_stocks_bottom_100['investment_date'].unique()},
    }

    valid_stock_ids = set(us_ret.columns)
    portfolio_ret = {}
    rebalance_dates = {}

    for portfolio_name in ['Top 100', f'Top {n_stocks_10}', 'Bottom 100']:
        portfolio_ret[portfolio_name], rebalance_dates[portfolio_name] = calculate_portfolio_returns(us_ret, selected_stocks[portfolio_name], valid_stock_ids)

    portfolio_ret['Naive'] = (1 + us_ret.mean(axis=1)).cumprod()
    portfolio_ret['SPY'] = benchmark_data['Cumulative Returns']

    portfolio_ret = pd.DataFrame(portfolio_ret)

    optimized_portfolios = {}
    portfolio_weights = {}
    for method in ['max_sharpe', 'min_variance', 'min_cvar']:
        optimized_returns, weights = process_portfolio(selected_stocks_top_100, us_ret, method, model, window_size)
        optimized_portfolios[method] = optimized_returns
        portfolio_weights[method] = weights

    plot_optimized_portfolios(portfolio_ret, optimized_portfolios, model, window_size, folder_name, rebalance_dates)

    portfolio_ret.to_csv(os.path.join(folder_name, f'portfolio_returns_{model}{window_size}.csv'))
    for method, returns in optimized_portfolios.items():
        returns.to_csv(os.path.join(folder_name, f'optimized_returns_{method}_{model}{window_size}.csv'))
    
    combined_returns = portfolio_ret.copy()
    for method, returns in optimized_portfolios.items():
        combined_returns[f'Optimized_{method}'] = (1 + returns).cumprod()
    
    combined_returns.to_csv(os.path.join(folder_name, f'combined_returns_{model}{window_size}.csv'))
    
    # Calculate and save performance metrics
    performance_metrics = calculate_performance_metrics(combined_returns, portfolio_weights, selected_stocks, n_stocks_10, us_ret)
    save_performance_metrics(performance_metrics, model, window_size, folder_name)
    
    return f"Completed processing for {model} with window size {window_size}"

def main():
    us_ret, removed_stocks = load_and_preprocess_data()
    logging.info("Data loaded and preprocessed.")

    benchmark_data = load_benchmark_data()
    logging.info("Benchmark data loaded.")

    for model in MODELS:
        for window_size in WINDOW_SIZES:
            result = process_model_window(model, window_size, us_ret, benchmark_data, removed_stocks)
            if result:
                logging.info(result)

if __name__ == "__main__":
    main()