# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

# 그래프 스타일 설정
import scienceplots
plt.style.use(['science', 'nature'])

# 설정 값들
TRAIN = '2018-12-31'
MODELS = ['CNN', 'TS']
WINDOW_SIZES = [5, 20, 60]
BASE_FOLDER = 'RIPT_WORK_SPACE/new_model_res/'

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

def load_and_process_ensemble_results(file_path, n_stocks, top=True):
    """
    앙상블 결과를 로드하고 처리하여 각 날짜별 상위 N개 또는 하위 N개 주식을 선택합니다.
    """
    # 앙상블 결과 로드
    ensemble_results = pd.read_feather(file_path)
    
    # up_prob 확인
    print(f"up_prob range: {ensemble_results['up_prob'].min()} to {ensemble_results['up_prob'].max()}")
    
    # Long format으로 변환 및 정렬
    ensemble_results_long = (ensemble_results
                             .pivot_table(index='ending_date', columns='StockID', values='up_prob')
                             .reset_index()
                             .melt(id_vars=['ending_date'], var_name='StockID', value_name='up_prob')
                             .sort_values(['ending_date', 'up_prob'], ascending=[True, not top]))
    
    # 각 날짜별 상위 N개 또는 하위 N개 주식 선택
    selected_stocks = ensemble_results_long.groupby('ending_date').head(n_stocks)
    selected_stocks.reset_index(drop=True, inplace=True)
    
    return selected_stocks

def select_top_stocks(top_stocks, rebalance_dates, n):
    """
    리밸런싱 날짜별로 상위 또는 하위 N개 주식의 리스트를 반환합니다.
    """
    selected_stocks = []
    for date in rebalance_dates:
        selected = top_stocks[top_stocks['ending_date'] == date]['StockID'].values
        selected_stocks.append((date, selected))
    return selected_stocks

def portfolio_optimization(returns, method='min_var'):
    """
    포트폴리오 최적화를 수행합니다. method는 'min_var' 또는 'max_sharpe'를 선택할 수 있습니다.
    """
    n = returns.shape[1]
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # 제약 조건: 가중치의 합은 1
    bounds = tuple((0, 1) for _ in range(n))  # 각 자산의 가중치는 0과 1 사이
    initial_weights = np.array(n * [1. / n])  # 초기 가중치는 균등 분배
    
    if method == 'min_var':
        # 포트폴리오 분산 최소화
        def portfolio_variance(weights):
            return weights.T @ cov_matrix @ weights
        optimized = minimize(portfolio_variance, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    elif method == 'max_sharpe':
        # 샤프 비율 최대화 (음수의 샤프 비율을 최소화)
        def negative_sharpe_ratio(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
            if portfolio_volatility == 0:
                return 0  # 변동성이 0인 경우 처리
            sharpe_ratio = portfolio_return / portfolio_volatility
            return -sharpe_ratio
        optimized = minimize(negative_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    else:
        raise ValueError("method should be 'min_var' or 'max_sharpe'")
    
    if optimized.success:
        return optimized.x
    else:
        print(f"Optimization failed: {optimized.message}")
        return initial_weights  # 최적화 실패 시 균등 가중치 반환

def calculate_portfolio_returns(us_ret, selected_stocks_list, method='min_var'):
    """
    포트폴리오 최적화 및 수익률 계산을 수행합니다.
    """
    portfolio_weights = {}
    
    for i, (rebalance_date, stock_ids) in enumerate(selected_stocks_list):
        # 리밸런싱 날짜 설정
        end_date = rebalance_date
        if i > 0:
            start_date = selected_stocks_list[i - 1][0]
        else:
            start_date = us_ret.index.min()
        
        # 최적화에 사용할 기간의 수익률 데이터 추출
        period_returns = us_ret.loc[start_date:end_date, stock_ids]
        
        # 결측치 처리
        period_returns = period_returns.dropna(axis=0, how='any')
        
        if period_returns.shape[0] < 2:
            print(f"Not enough data to optimize for period ending on {end_date}")
            continue
        
        # 포트폴리오 최적화 수행
        try:
            if method == 'equal_weight':
                n = len(stock_ids)
                weights = np.array(n * [1. / n])
            else:
                weights = portfolio_optimization(period_returns, method=method)
            portfolio_weights[rebalance_date] = pd.Series(weights, index=stock_ids)
        except Exception as e:
            print(f"Optimization failed for period starting on {rebalance_date}: {e}")
            continue
    
    # 포트폴리오 수익률 계산
    portfolio_returns = pd.Series(dtype=float)
    
    dates = list(portfolio_weights.keys())
    for i, date in enumerate(dates):
        weights_series = portfolio_weights[date]
        start_date = date
        if i < len(dates) - 1:
            end_date = dates[i + 1]
        else:
            end_date = us_ret.index.max()
        
        # 투자 기간의 수익률 데이터 추출
        period_returns = us_ret.loc[start_date:end_date, weights_series.index]
        # 일별 포트폴리오 수익률 계산
        daily_portfolio_returns = period_returns.dot(weights_series)
        # 포트폴리오 수익률에 추가
        portfolio_returns = portfolio_returns._append(daily_portfolio_returns)
    
    # 누적 수익률 계산
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    return cumulative_returns

def calculate_portfolio_up_prob(selected_stocks, rebalance_dates):
    """
    선택된 주식들의 평균 up_prob를 계산합니다.
    """
    up_prob_series = pd.Series(dtype=float)
    for date in rebalance_dates:
        stocks_at_date = selected_stocks[selected_stocks['ending_date'] == date]
        avg_up_prob = stocks_at_date['up_prob'].mean()
        up_prob_series[date] = avg_up_prob
    return up_prob_series

def main():
    # us_ret 데이터 로드
    us_ret = pd.read_feather('./RIPT_processed_data/us_ret_TRAIN.feather')
    
    # 모델 및 윈도우 사이즈 설정
    models = ['CNN', 'TS']
    window_sizes = [5, 20, 60]
    
    # 포트폴리오 선택 방법
    PORTFOLIO_N = [20, 50]
    NAIVE_N = us_ret.shape[1]  # 전체 주식 수
    
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
            f'Top {N} (min_var)': {'n_stocks': N, 'top': True, 'method': 'min_var'},
            f'Top {N} (max_sharpe)': {'n_stocks': N, 'top': True, 'method': 'max_sharpe'},
            f'Top {N} (Equal Weight)': {'n_stocks': N, 'top': True, 'method': 'equal_weight'},
            f'Bottom {N} (Equal Weight)': {'n_stocks': N, 'top': False, 'method': 'equal_weight'},
        })
    
    # 나이브 포트폴리오 추가
    portfolio_selections['Naive'] = {'n_stocks': NAIVE_N, 'top': True, 'method': 'equal_weight'}
    
    for model in models:
        for window_size in window_sizes:
            # 앙상블 결과 파일 로드
            file_path = f'./RIPT_WORK_SPACE/ensemble_{model}{window_size}_res.feather'
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
            print(f'Processing Model: {model}, Window Size: {window_size}')
            
            cumulative_returns_dict = {}  # 포트폴리오별 누적 수익률 저장
            up_prob_dict = {}  # 포트폴리오별 평균 up_prob 저장
            
            for selection_name, selection_params in portfolio_selections.items():
                n_stocks = selection_params['n_stocks']
                top = selection_params['top']
                method = selection_params.get('method', 'equal_weight')
                
                # 선택한 주식 로드
                selected_stocks = load_and_process_ensemble_results(file_path, n_stocks=n_stocks, top=top)
                
                # 리밸런싱 날짜 설정
                selected_stocks = selected_stocks[selected_stocks['ending_date'] >= TRAIN]
                rebalance_dates = selected_stocks['ending_date'].unique()
                
                # 포트폴리오 구성
                selected_stocks_list = select_top_stocks(selected_stocks, rebalance_dates, n_stocks)
                
                # 포트폴리오 최적화 및 수익률 계산
                cumulative_returns = calculate_portfolio_returns(us_ret, selected_stocks_list, method=method)
                cumulative_returns_dict[selection_name] = cumulative_returns
                
                # 평균 up_prob 계산
                up_prob_series = calculate_portfolio_up_prob(selected_stocks, rebalance_dates)
                up_prob_dict[selection_name] = up_prob_series

            
            # 누적 수익률을 하나의 데이터프레임으로 결합
            cumulative_returns_df = pd.DataFrame(cumulative_returns_dict)
            up_prob_df = pd.DataFrame(up_prob_dict)
            
            # 평균 up_prob 출력
            print(f"\nAverage up_prob for each portfolio - Model: {model}, Window Size: {window_size}")
            print(up_prob_df.mean())
            
            # 그래프 그리기
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            fig3, ax3 = plt.subplots(figsize=(12, 8))

            # 누적 수익률 그래프 (로그 스케일)
            color_index = 0
            for label, cumulative_returns in cumulative_returns_dict.items():
                if label == 'Benchmark (Top 50 Optimized)':
                    color = benchmark_color
                    linestyle = '-.'
                    linewidth = 2
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
                
                if 'min_var' in label or 'max_sharpe' in label:
                    linewidth = 1
                
                ax1.semilogy(cumulative_returns.index, cumulative_returns.values, label=label, color=color, linestyle=linestyle, linewidth=linewidth)

            ax1.set_title(f'Cumulative Returns (Log Scale) - Model: {model}, Window Size: {window_size}')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Cumulative Returns (Log Scale)')
            ax1.legend()
            ax1.grid(True)
            fig1.tight_layout()
            fig1.savefig(f'cumulative_returns_{model}{window_size}.png')

            # 예측 편향 검증
            benchmark = cumulative_returns_df.mean(axis=1)
            relative_performance = cumulative_returns_df.div(benchmark, axis=0)

            color_index = 0
            for label, rel_perf in relative_performance.items():
                if label == 'Benchmark 1/N':
                    continue  # 벤치마크 자체는 그리지 않음
                elif 'Bottom' in label:
                    color = get_color(color_index, True)
                    linestyle = '--'
                elif 'Top' in label:
                    color = get_color(color_index, False)
                    linestyle = '-'
                    color_index += 1
                else:  # Naive
                    color = neutral_color
                    linestyle = '-'
                
                ax2.plot(rel_perf.index, rel_perf.values, label=label, color=color, linestyle=linestyle)

            ax2.set_title(f'Relative Performance (Prediction Bias Check) - Model: {model}, Window Size: {window_size}')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Relative Performance to Benchmark')
            ax2.legend()
            ax2.grid(True)
            ax2.axhline(y=1, color='black', linestyle='--')  # 벤치마크 라인
            fig2.tight_layout()
            fig2.savefig(f'relative_performance_{model}{window_size}.png')

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
                
                ax3.plot(up_prob_series.index, up_prob_series.values, label=label, color=color, linestyle=linestyle)

            ax3.set_title(f'Average up_prob - Model: {model}, Window Size: {window_size}')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Average up_prob')
            ax3.legend()@
            ax3.grid(True)
            fig3.tight_layout()
            fig3.savefig(f'average_up_prob_{model}{window_size}.png')

            # 폴더 생성
            folder_name = f'{model}_{window_size}'
            os.makedirs(folder_name, exist_ok=True)

            # 그래프 저장
            fig1.savefig(os.path.join(folder_name, f'cumulative_returns_{model}{window_size}.png'))
            fig2.savefig(os.path.join(folder_name, f'relative_performance_{model}{window_size}.png'))
            fig3.savefig(os.path.join(folder_name, f'average_up_prob_{model}{window_size}.png'))

            plt.close('all')

            # CSV 파일 저장
            cumulative_returns_df.to_csv(os.path.join(folder_name, f'cumulative_returns_{model}{window_size}.csv'), index=True)
            relative_performance.to_csv(os.path.join(folder_name, f'relative_performance_{model}{window_size}.csv'), index=True)
            up_prob_df.to_csv(os.path.join(folder_name, f'up_prob_{model}{window_size}.csv'), index=True)
            
if __name__ == "__main__":

    main()
