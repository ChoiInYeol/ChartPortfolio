# coding: utf-8
# optimization.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

def _ret_to_cum_log_ret(rets):
    log_rets = np.log(rets.astype(float) + 1)
    return log_rets.cumsum()

# 그래프 스타일 설정
import scienceplots
plt.style.use(['science'])

# 설정 값들
TRAIN = '2017-12-31'
MODELS = ['CNN']
WINDOW_SIZES = [5, 20]
BASE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'WORK_DIR')

# 1. 데이터 로드 및 전처리
def load_and_preprocess_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'processed_data', 'us_ret.feather')
    us_ret = pd.read_feather(file_path)
    us_ret = us_ret[us_ret['Date'] >= TRAIN]
    us_ret = us_ret.pivot(index='Date', columns='StockID', values='Ret')
    us_ret.index = pd.to_datetime(us_ret.index)

    # Check if returns are in percentages and convert to decimals if necessary
    if us_ret.abs().mean().mean() > 1:
        print("Converting returns from percentages to decimals.")
        us_ret = us_ret / 100

    # 주가 데이터 로드
    stock_prices_path = os.path.join(current_dir, 'processed_data', 'us_ret.feather')
    stock_prices = pd.read_feather(stock_prices_path)
    stock_prices = stock_prices.pivot(index='Date', columns='StockID', values='Close')
    stock_prices.index = pd.to_datetime(stock_prices.index)
    stock_prices = stock_prices.fillna(method='ffill')  # 결측치 채움

    # 이상치 처리 함수 호출
    us_ret_cleaned = handle_outliers(us_ret, stock_prices)

    return us_ret_cleaned

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
        pd.DataFrame: 이상치가 처리된 수익률 데이터.
    """
    high_threshold = 0.75  # 75% 이상의 수익률   
    low_threshold = -0.75  # -75% 이하의 수익률

    # 1. 극단적인 수익률을 가진 주식 탐지
    extreme_returns = returns[(returns > high_threshold) | (returns < low_threshold)]

    # 2. 변동성이 높은 주식 식별 (최대 변동률이 volatility_threshold 이상인 주식)
    max_daily_volatility = prices.pct_change().abs().max()
    stocks_to_remove = max_daily_volatility[max_daily_volatility > volatility_threshold].index

    # 주식 제거 결과 출력
    print(f"총 {len(stocks_to_remove)}개의 주식이 변동성 기준을 초과하여 제거됩니다.")
    print(f"제거된 주식 목록: {list(stocks_to_remove)}")

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
    print(f"총 {adjusted_count}개의 이상치 수익률을 조정했습니다.")
    print(f"총 {adjusted_stocks}개의 주식에 대해 수익률 조정을 수행했습니다.")

    return returns_cleaned

def load_benchmark_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'processed_data', 'snp500_index.csv')
    snp500 = pd.read_csv(file_path, parse_dates=['Date'])
    snp500.set_index('Date', inplace=True)
    snp500.sort_index(inplace=True)
    snp500 = snp500[snp500.index >= TRAIN]
    # Use 'Adj Close' as adjusted closing price
    snp500['Returns'] = snp500['Adj Close'].pct_change()
    # Calculate cumulative returns
    snp500['Cumulative Returns'] = (1 + snp500['Returns']).cumprod()
    return snp500

# 2. 앙상블 결과 처리
def load_ensemble_results(folder_path):
    """
    폴더 내의 모든 앙상블 결과 파일을 로드하고 하나의 데이터프레임으로 병합합니다.
    """
    all_data = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv') and 'ensem' in file:
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            
            df['ending_date'] = pd.to_datetime(df['ending_date'])
            df['investment_date'] = df['ending_date'] - pd.DateOffset(months=1)
            df['StockID'] = df['StockID'].astype(str)
            df.set_index(['investment_date', 'StockID'], inplace=True)
            
            all_data.append(df)
        
    if not all_data:
        raise ValueError(f"No CSV files found in {folder_path}")
    
    combined_df = pd.concat(all_data)
    combined_df.sort_index(inplace=True)
    
    # unnamed 컬럼 제거
    combined_df = combined_df.loc[:, ~combined_df.columns.str.contains('^Unnamed')]
    
    # MarketCap 컬럼 제거
    if 'MarketCap' in combined_df.columns:
        combined_df.drop('MarketCap', axis=1, inplace=True)
    
    return combined_df

def process_ensemble_results(base_folder):
    for model in MODELS:
        for w in WINDOW_SIZES:
            folder_path = os.path.join(base_folder, f'{model}{w}', f'{w}D20P', 'ensem_res')
            output_path = os.path.join(base_folder, f'ensemble_{model}{w}_res.feather')
            
            print(f'Processing Ensemble {model}{w}...   ', end='')
            try:
                ensemble_results = load_ensemble_results(folder_path)
                ensemble_results.reset_index().to_feather(output_path)
                print(f'Shape of the dataframe: {ensemble_results.shape}')
            except Exception as e:
                print(f'Error processing {model}{w}: {str(e)}')

def load_and_process_ensemble_results(file_path, n_stocks, top=True):
    """
    앙상블 결과를 로드하고 처리합니다.

    Args:
        file_path (str): 앙상블 결과 파일 경로
        n_stocks (int): 선택할 주식 수
        top (bool): True면 상위 주식 선택, False면 하위 주식 선택

    Returns:
        pd.DataFrame: 처리된 주식 선택 결과
    """
    # 앙상블 결과 로드
    ensemble_results = pd.read_feather(file_path)
    
    # MultiIndex인 경우 리셋
    if isinstance(ensemble_results.index, pd.MultiIndex):
        ensemble_results = ensemble_results.reset_index()
    
    # 날짜 변환
    ensemble_results['investment_date'] = pd.to_datetime(ensemble_results['investment_date'])
    ensemble_results['ending_date'] = pd.to_datetime(ensemble_results['ending_date'])
    
    # StockID를 문자열로 변환
    ensemble_results['StockID'] = ensemble_results['StockID'].astype(str)
    
    # investment_date 기준으로 정렬 및 up_prob에 따라 선택
    ensemble_results_sorted = ensemble_results.sort_values(['investment_date', 'up_prob'], ascending=[True, not top])
    
    # 각 investment_date에 대해 상위 또는 하위 N개 주식 선택
    selected_stocks = ensemble_results_sorted.groupby('investment_date').head(n_stocks)
    selected_stocks.reset_index(drop=True, inplace=True)
    
    return selected_stocks

def select_top_stocks(selected_stocks, rebalance_dates, n):
    """
    리밸런싱 날짜별로 상위 또는 하위 N개 주식의 리스트를 반환합니다.
    """
    selected_stocks_list = []
    for date in rebalance_dates:
        stocks_at_date = selected_stocks[selected_stocks['investment_date'] == date]
        selected = stocks_at_date['StockID'].values
        selected_stocks_list.append((date, selected))
    return selected_stocks_list

def get_next_valid_date(date, date_index):
    while date not in date_index and date <= date_index.max():
        date += pd.Timedelta(days=1)
    return date if date in date_index else None

def get_previous_valid_date(date, date_index):
    while date not in date_index and date >= date_index.min():
        date -= pd.Timedelta(days=1)
    return date if date in date_index else None

def filter_invalid_stocks(selected_stocks_list, valid_stock_ids):
    """
    이미 이상치로 걸러진 주식들을 선택된 포트폴리오에서 제외합니다.

    Args:
        selected_stocks_list (list): (rebalance_date, stock_ids)의 리스트.
        valid_stock_ids (set): 유효한 주식 ID의 집합.

    Returns:
        list: 유효한 주식만 포함된 (rebalance_date, stock_ids)의 리스트.
    """
    filtered_list = []
    for rebalance_date, stock_ids in selected_stocks_list:
        # 유효한 주식만 필터링
        filtered_stock_ids = [stock_id for stock_id in stock_ids if stock_id in valid_stock_ids]
        if filtered_stock_ids:  # 유효한 주식이 있으면 추가
            filtered_list.append((rebalance_date, filtered_stock_ids))
    return filtered_list

def calculate_portfolio_returns(us_ret, selected_stocks_list, valid_stock_ids, method='equal_weight'):
    """
    선택된 포트폴리오의 수익률을 계산합니다.
    
    Args:
        us_ret (pd.DataFrame): 수익률 데이터.
        selected_stocks_list (list): 선택된 (rebalance_date, stock_ids) 리스트.
        valid_stock_ids (set): 유효한 주식 ID의 집합.
        method (str): 'equal_weight'를 기본으로 사용하는 수익률 계산 방법.
        
    Returns:
        pd.Series: 누적 수익률 시리즈.
    """
    # 선택된 주식 목록에서 유효한 주식만 필터링
    selected_stocks_list = filter_invalid_stocks(selected_stocks_list, valid_stock_ids)

    portfolio_returns = pd.Series(dtype=float)

    for i, (investment_date, stock_ids) in enumerate(selected_stocks_list):
        # 투자 시작일은 investment_date의 다음 거래일
        start_date = get_next_valid_date(investment_date + pd.Timedelta(days=1), us_ret.index)
        if start_date is None:
            continue  # 유효한 시작 날짜가 없으면 건너뜀

        # 투자 종료일 계산
        if i < len(selected_stocks_list) - 1:
            next_investment_date = selected_stocks_list[i + 1][0]
            end_date = get_previous_valid_date(next_investment_date, us_ret.index)
        else:
            end_date = us_ret.index.max()

        if end_date < start_date:
            continue  # 유효하지 않은 날짜 범위 건너뜀

        # 투자 기간의 수익률 가져오기
        period_returns = us_ret.loc[start_date:end_date, stock_ids]
        period_returns = period_returns.dropna(axis=1, how='any')
        if period_returns.empty:
            continue

        # 포트폴리오 비중 결정
        num_stocks = period_returns.shape[1]
        weights = np.array([1 / num_stocks] * num_stocks)

        # 일별 포트폴리오 수익률 계산
        daily_portfolio_returns = period_returns.dot(weights)

        # 결과를 포트폴리오 수익률 시리즈에 추가
        portfolio_returns = portfolio_returns.add(daily_portfolio_returns, fill_value=0)

    # 날짜별로 정렬
    portfolio_returns = portfolio_returns.sort_index()

    # 누적 수익률 계산
    cumulative_returns = (1 + portfolio_returns).cumprod()

    return cumulative_returns

def calculate_portfolio_up_prob(selected_stocks, us_ret):
    """
    투자 기간 동안 평균 up_prob를 유지하여 시계열로 반환합니다.
    """
    up_prob_series = pd.Series(dtype=float)
    for date in selected_stocks['investment_date'].unique():
        avg_up_prob = selected_stocks[selected_stocks['investment_date'] == date]['up_prob'].mean()
        
        # 해당 투자 기간의 날짜들 가져오기
        start_date = get_next_valid_date(date + pd.Timedelta(days=1), us_ret.index)
        if start_date is None:
            continue  # No valid start date
        
        next_dates = selected_stocks[selected_stocks['investment_date'] > date]['investment_date']
        if not next_dates.empty:
            end_date = get_previous_valid_date(next_dates.min(), us_ret.index)
        else:
            end_date = us_ret.index.max()
        
        if end_date < start_date:
            continue
        
        date_range = us_ret.loc[start_date:end_date].index
        temp_series = pd.Series(avg_up_prob, index=date_range)
        if not temp_series.empty:
            up_prob_series = pd.concat([up_prob_series, temp_series])
    
    up_prob_series = up_prob_series.sort_index()
    return up_prob_series

def make_portfolio_plot(portfolio_ret, model, window_size, result_dir):
    # 누적 로그 수익률 계산
    log_ret_df = pd.DataFrame(index=portfolio_ret.index)
    for column in portfolio_ret.columns:
        log_ret_df[column] = _ret_to_cum_log_ret(portfolio_ret[column])
    
    # 모든 시리즈가 0부터 시작하도록 조정
    prev_year = pd.to_datetime(log_ret_df.index[0]).year - 1
    prev_day = pd.to_datetime(f"{prev_year}-12-31")
    log_ret_df.loc[prev_day] = 0
    log_ret_df = log_ret_df.sort_index()
    
    # 그래프 그리기
    columns_to_plot = ["Top", "Bottom", "Top-Bottom", "Naive", "Benchmark"]
    colors = {"Top": "b", "Bottom": "r", "Top-Bottom": "k", "Naive": "y", "Benchmark": "g"}
    
    plot = log_ret_df[columns_to_plot].plot(
        style=colors,
        lw=1,
        title=f'{model} Model (Window Size: {window_size})',
        figsize=(10, 6)
    )
    plot.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 저장
    save_path = os.path.join(result_dir, f'{model}_window{window_size}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Portfolio plot saved to {save_path}")

def main():
    
    # 설정 값들
    TRAIN = '2017-12-31'
    MODELS = ['CNN']
    WINDOW_SIZES = [5, 20]
    BASE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'WORK_DIR')
    
    # 1. 데이터 로드 및 전처리
    us_ret = load_and_preprocess_data()
    print("Data loaded and preprocessed.")

    # 벤치마크 데이터 로드
    benchmark_data = load_benchmark_data()
    print("Benchmark data loaded.")

    # 2. 앙상블 결과 처리
    process_ensemble_results(BASE_FOLDER)
    print("Ensemble results processed.")
    
    # 모델 및 윈도우 사이즈 설정
    models = MODELS
    window_sizes = WINDOW_SIZES
    
    # 포트폴리오 선택 방법
    NAIVE_N = us_ret.shape[1]  # 전체 주식 수
    PORTFOLIO_N = [round(NAIVE_N / 40), round(NAIVE_N / 20), round(NAIVE_N / 2)]
    
    
    # 색상 설정
    colors = [
        '#FF4136',  # 빨간색 (Top)
        '#0074D9',  # 파란색 (Bottom)
        '#FF851B',  # 주황색 (Top)
        '#39CCCC',  # 청록색 (Bottom)
        '#B10DC9',  # 보라색 (Top)
        '#3D9970',  # 초록색 (Bottom)
        '#FF4136',  # 빨간색 (Top, 반복)
        '#0074D9',  # 파란색 (Bottom, 반복)
        '#FF851B',  # 주황색 (Top, 반복)
        '#39CCCC',  # 청록색 (Bottom, 반복)
    ]
    neutral_color = '#AAAAAA'  # 회색 (Naive 포트폴리오용)
    benchmark_color = '#2ECC40'  # 밝은 초록색 (벤치마크용)

    def get_color(index, is_bottom):
        color_index = index % (len(colors) // 2)
        return colors[color_index * 2 + (1 if is_bottom else 0)]

    portfolio_selections = {}
    for N in PORTFOLIO_N:
        portfolio_selections.update({
            f'Top {N} (Equal Weight)': {'n_stocks': N, 'top': True, 'method': 'equal_weight'},
            f'Bottom {N} (Equal Weight)': {'n_stocks': N, 'top': False, 'method': 'equal_weight'},
        })
    
    # 나이브 포트폴리오 추가
    portfolio_selections['Naive'] = {'n_stocks': NAIVE_N, 'top': True, 'method': 'equal_weight'}
    
    for model in models:
        for window_size in window_sizes:
            # 앙상블 결과 파일 로드
            file_path = os.path.join(BASE_FOLDER, f'ensemble_{model}{window_size}_res.feather')
            if not os.path.exists(file_path):
                print(f"파일을 찾을 수 없습니다: {file_path}")
                continue
            print(f'처리 중인 모델: {model}, 윈도우 크기: {window_size}')
            
            cumulative_returns_dict = {}  # 포트폴리오별 누적 수익률 저장
            up_prob_dict = {}  # 포트폴리오별 평균 up_prob 저장

            # 벤치마크 추가
            cumulative_returns_dict['Benchmark'] = benchmark_data['Cumulative Returns']

            # 폴더 생성
            folder_name = os.path.join(BASE_FOLDER, f'{model}{window_size}')
            os.makedirs(folder_name, exist_ok=True)

            for selection_name, selection_params in portfolio_selections.items():
                n_stocks = selection_params['n_stocks']
                top = selection_params['top']
                method = selection_params.get('method', 'equal_weight')
                
                # 선택한 주식 로드
                selected_stocks = load_and_process_ensemble_results(file_path, n_stocks=n_stocks, top=top)
                
                # 리밸런싱 날짜 설정
                selected_stocks = selected_stocks[selected_stocks['investment_date'] >= TRAIN]
                rebalance_dates = selected_stocks['investment_date'].unique()
                
                # 포트폴리오 구성
                selected_stocks_list = select_top_stocks(selected_stocks, rebalance_dates, n_stocks)
                
                # 포트폴리오 수익률 계산
                valid_stock_ids = set(us_ret.columns)  # 유효한 주식 ID 집합 생성
                cumulative_returns = calculate_portfolio_returns(us_ret, selected_stocks_list, valid_stock_ids)
                cumulative_returns_dict[selection_name] = cumulative_returns
                
                # 평균 up_prob 계산
                up_prob_series = calculate_portfolio_up_prob(selected_stocks, us_ret)
                up_prob_dict[selection_name] = up_prob_series

            # 포트폴리오 수익률 계산 (기존 코드에서 가져옴)
            portfolio_ret = pd.DataFrame({
                'Top': cumulative_returns_dict[f'Top {PORTFOLIO_N[0]} (Equal Weight)'],
                'Bottom': cumulative_returns_dict[f'Bottom {PORTFOLIO_N[0]} (Equal Weight)'],
                'Naive': cumulative_returns_dict['Naive'],
                'Benchmark': cumulative_returns_dict['Benchmark']
            })
            portfolio_ret['Top-Bottom'] = portfolio_ret['Top'] - portfolio_ret['Bottom']

            # 그래프 생성
            make_portfolio_plot(portfolio_ret, model, window_size, folder_name)

            # 누적 수익률을 하나의 데이터프레임으로 결합
            cumulative_returns_df = pd.DataFrame(cumulative_returns_dict)
            up_prob_df = pd.DataFrame(up_prob_dict)
            
            # 평균 up_prob 출력
            print(f"\nAverage up_prob for each portfolio - Model: {model}, Window Size: {window_size}")
            print(up_prob_df.mean())
            
            # 그래프 그리기 부분 수정
            fig1, ax1 = plt.subplots(figsize=(8, 7), dpi=600)
            fig2, ax2 = plt.subplots(figsize=(8, 7), dpi=600)
            fig3, ax3 = plt.subplots(figsize=(8, 7), dpi=600)

            # 누적 수익률 그래프
            color_index = 0
            for label, cumulative_returns in cumulative_returns_dict.items():
                if label == 'Benchmark':
                    color = benchmark_color
                    linestyle = '-'
                    linewidth = 1.5
                elif 'Bottom' in label:
                    color = get_color(color_index, True)
                    linestyle = '--'
                    linewidth = 1
                elif 'Top' in label:
                    color = get_color(color_index, False)
                    linestyle = '-'
                    linewidth = 1
                    color_index += 1
                else:  # Naive
                    color = neutral_color
                    linestyle = '-'
                    linewidth = 1
                
                ax1.plot(cumulative_returns.index, cumulative_returns.values, label=label, color=color, linestyle=linestyle, linewidth=linewidth)
                
                # 리밸런싱 날짜에 점 추가 (마커 크기 축소)
                if label != 'Benchmark':
                    for date in cumulative_returns.index.intersection(rebalance_dates):
                        ax1.plot(date, cumulative_returns.loc[date], marker='o', color=color, markersize=3)

            ax1.set_title(f'Cumulative Returns - {model} {window_size}-day', fontsize=10)
            ax1.set_xlabel('Date', fontsize=8)
            ax1.set_ylabel('Cumulative Returns', fontsize=8)
            ax1.legend(fontsize=6, ncol=2)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='both', which='major', labelsize=6)
            fig1.tight_layout()

            # 예측 편향 검증
            benchmark = cumulative_returns_df['Naive']
            relative_performance = cumulative_returns_df.div(benchmark, axis=0)

            color_index = 0
            for label, rel_perf in relative_performance.items():
                if label == 'Naive':
                    continue  # 벤치마크 자체는 그리지 않음
                elif 'Bottom' in label:
                    color = get_color(color_index, True)
                    linestyle = '--'
                elif 'Top' in label:
                    color = get_color(color_index, False)
                    linestyle = '-'
                    color_index += 1
                else:  # Benchmark
                    color = benchmark_color
                    linestyle = '-'
                
                ax2.plot(rel_perf.index, rel_perf.values, label=label, color=color, linestyle=linestyle, linewidth=1)

            ax2.set_title(f'Relative Performance - {model} {window_size}-day', fontsize=10)
            ax2.set_xlabel('Date', fontsize=8)
            ax2.set_ylabel('Relative Performance to Naive', fontsize=8)
            ax2.legend(fontsize=6, ncol=2)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=1, color='black', linestyle='--', linewidth=0.5)  # 벤치마크 라인
            ax2.tick_params(axis='both', which='major', labelsize=6)
            fig2.tight_layout()

            # 평균 up_prob 그래프
            color_index = 0
            for label, up_prob_series in up_prob_dict.items():
                if 'Bottom' in label:
                    color = get_color(color_index, True)
                    linestyle = '--'
                elif 'Top' in label:
                    color = get_color(color_index, False)
                    linestyle = '-'
                    color_index += 1
                else:  # Naive
                    color = neutral_color
                    linestyle = '-'
                
                ax3.plot(up_prob_series.index, up_prob_series.values, label=label, color=color, linestyle=linestyle, linewidth=1)

            ax3.set_title(f'Average up_prob - {model} {window_size}-day', fontsize=10)
            ax3.set_xlabel('Date', fontsize=8)
            ax3.set_ylabel('Average up_prob', fontsize=8)
            ax3.legend(fontsize=6, ncol=2)
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='both', which='major', labelsize=6)
            fig3.tight_layout()

            # 그래프 저장
            fig1.savefig(os.path.join(folder_name, f'cumulative_returns_{model}{window_size}.png'), dpi=600, bbox_inches='tight')
            fig2.savefig(os.path.join(folder_name, f'relative_performance_{model}{window_size}.png'), dpi=600, bbox_inches='tight')
            fig3.savefig(os.path.join(folder_name, f'average_up_prob_{model}{window_size}.png'), dpi=600, bbox_inches='tight')

            plt.close('all')

            # CSV 파일 저장
            cumulative_returns_df.to_csv(os.path.join(folder_name, f'cumulative_returns_{model}{window_size}.csv'), index=True)
            relative_performance.to_csv(os.path.join(folder_name, f'relative_performance_{model}{window_size}.csv'), index=True)
            up_prob_df.to_csv(os.path.join(folder_name, f'up_prob_{model}{window_size}.csv'), index=True)
            
if __name__ == "__main__":
    main()