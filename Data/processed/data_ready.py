"""
Prepare and process downloaded stock data.
This script handles tasks 5-7 of the data preparation process.
"""

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from scipy import stats
import FinanceDataReader as fdr
import inquirer
from pathlib import Path

def setup_logging(log_dir='logs'):
    """Initialize logging configuration"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'stock_filter_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Logging started")
    return log_file

def download_spy():
    """Download SPY ETF data"""
    try:
        spy_data = fdr.DataReader('S&P500')
        spy_data.index.name = 'Date'
        spy_data.to_csv('snp500_index.csv')
        logging.info("SPY data download completed")
    except Exception as e:
        logging.error(f"SPY data download failed: {e}")

def filter_stocks(
    input_file, 
    output_file, 
    min_trading_days=1000,
    start_date='2001-01-01', 
    end_date='2024-08-01',
):
    """
    Filter stock data using t-distribution to remove outliers.
    
    Parameters
    ----------
    input_file : str
        Path to input CSV file containing stock data
    output_file : str
        Path to save filtered stock data
    min_trading_days : int, default 1000
        Minimum number of trading days required
    start_date : str, default '2001-01-01'
        Start date for filtering in 'YYYY-MM-DD' format
    end_date : str, default '2024-08-01'
        End date for filtering in 'YYYY-MM-DD' format
        
    Returns
    -------
    pd.DataFrame
        Filtered stock data
    """
    
    logging.info("Starting stock data filtering process")
    
    try:
        # Load data
        df = pd.read_csv(input_file)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        
        # Filter date range
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        # Calculate trading days per stock
        trading_days = df.groupby('PERMNO').size()
        valid_stocks = trading_days[trading_days >= min_trading_days].index
        df = df[df['PERMNO'].isin(valid_stocks)]
        
        logging.info(f"Stocks with sufficient trading days: {len(valid_stocks)}")
        
        # Calculate statistics for each stock
        stock_stats = []
        for permno in valid_stocks:
            stock_data = df[df['PERMNO'] == permno]
            
            stats_dict = {
                'PERMNO': permno,
                'mean_ret': stock_data['RET'].mean(),
                'std_ret': stock_data['RET'].std(),
                'skew_ret': stats.skew(stock_data['RET']),
                'kurt_ret': stats.kurtosis(stock_data['RET']),
                'trading_days': len(stock_data)
            }
            stock_stats.append(stats_dict)
            
        stats_df = pd.DataFrame(stock_stats)
        
        # Calculate t-statistics for each metric
        metrics = ['mean_ret', 'std_ret', 'skew_ret', 'kurt_ret']
        t_stats = {}
        
        for metric in metrics:
            t_stats[metric] = np.abs(stats.zscore(stats_df[metric]))
        
        # Filter stocks based on t-statistics (95% confidence interval)
        valid_stocks = set(stats_df['PERMNO'])
        for metric in metrics:
            extreme_stocks = stats_df[t_stats[metric] > 1.96]['PERMNO']
            valid_stocks -= set(extreme_stocks)
            logging.info(f"Removed {len(extreme_stocks)} stocks based on {metric}")
        
        # Create final filtered dataset
        filtered_df = df[df['PERMNO'].isin(valid_stocks)].copy()
        
        # Sort by PERMNO and date
        filtered_df.sort_values(['PERMNO', 'date'], inplace=True)
        
        # Save filtered data
        filtered_df.to_csv(output_file, index=False)
        
        logging.info(f"Filtering complete. Remaining stocks: {len(valid_stocks)}")
        logging.info(f"Filtered data saved to {output_file}")
        
        return filtered_df
        
    except Exception as e:
        logging.error(f"Error in filter_stocks: {str(e)}")
        raise

def create_SP500_return_df():
    """
    Create a dataframe of returns for SP500 2018 constituents.
    
    Returns:
        pd.DataFrame: 인덱스는 날짜, 컬럼은 종목코드인 수익률 데이터프레임
    """
    
    # 파일 경로 설정
    data_dir = Path(__file__).parent
    sp500_path = data_dir / 'sp500_20180101.csv'
    symbol_permno_path = data_dir / 'symbol_permno.csv'
    filtered_stocks_path = data_dir / 'filtered_stock.csv'
    
    # SP500 2018년 구성종목 로드
    sp500_tickers = pd.read_csv(sp500_path)['Symbol'].tolist()
    
    # Symbol-PERMNO 매핑 로드
    symbol_permno = pd.read_csv(symbol_permno_path)
    
    # SP500 종목들의 PERMNO 값 추출
    sp500_permnos = symbol_permno[symbol_permno['Symbol'].isin(sp500_tickers)]['PERMNO'].tolist()
    
    # filtered_stock.csv에서 수익률 데이터 로드
    stocks_df = pd.read_csv(filtered_stocks_path, parse_dates=['date'])
    
    # PERMNO를 Symbol로 변환하기 위한 매핑 딕셔너리 생성
    permno_to_symbol = dict(zip(symbol_permno['PERMNO'], symbol_permno['Symbol']))
    
    # SP500 구성종목만 필터링 (PERMNO 기준) 및 복사본 생성
    target_stocks = stocks_df[stocks_df['PERMNO'].isin(sp500_permnos)].copy()
    
    # PERMNO를 Symbol로 변환
    target_stocks.loc[:, 'Symbol'] = target_stocks['PERMNO'].map(permno_to_symbol)
    
    # 수익률 데이터를 피벗 테이블로 변환
    return_df = target_stocks.pivot(
        index='date',
        columns='Symbol',
        values='RET'
    )
    
    # 결과 저장
    return_df.to_csv(data_dir / 'sp500_return_df.csv')
    print(f"수익률 데이터 생성 완료: {return_df.shape}")
    print(f"기간: {return_df.index.min()} ~ {return_df.index.max()}")
    print(f"종목 수: {len(return_df.columns)}")
    
    return return_df

def create_return_df():
    """
    Create a dataframe of returns for filtered stocks.
    Returns:
        pd.DataFrame: 인덱스는 날짜, 컬럼은 종목코드인 수익률 데이터프레임
    """
    
    # 파일 경로 설정
    data_dir = Path(__file__).parent
    symbol_permno_path = data_dir / 'symbol_permno.csv'
    filtered_stocks_path = data_dir / 'filtered_stock.csv'
    
    # Symbol-PERMNO 매핑 로드
    symbol_permno = pd.read_csv(symbol_permno_path)

    # filtered_stock.csv에서 수익률 데이터 로드
    stocks_df = pd.read_csv(filtered_stocks_path, parse_dates=['date'])
    
    # PERMNO를 Symbol로 변환하기 위한 매핑 딕셔너리 생성
    permno_to_symbol = dict(zip(symbol_permno['PERMNO'], symbol_permno['Symbol']))
    
    # PERMNO를 Symbol로 변환
    stocks_df.loc[:, 'Symbol'] = stocks_df['PERMNO'].map(permno_to_symbol)
    
    # 수익률 데이터를 피벗 테이블로 변환
    return_df = stocks_df.pivot(
        index='date',
        columns='Symbol',
        values='RET'
    )
    
    # 결과 저장
    return_df.to_csv(data_dir / 'return_df.csv')
    print(f"수익률 데이터 생성 완료: {return_df.shape}")
    print(f"기간: {return_df.index.min()} ~ {return_df.index.max()}")
    print(f"종목 수: {len(return_df.columns)}")
    
    return return_df

def main():
    """Main execution function for data preparation"""
    tasks = [
        ("1. Download SPY ETF data", "download_spy"),
        ("2. Filter outlier data (create filtered_stock.csv)", "filter_abnormal"),
        ("3. Create return dataframe (create return_df.csv)", "create_return_df"),
        ("4. Create SP500 return dataframe (create sp500_return_df.csv)", "create_SP500_return_df")
    ]
    
    questions = [
        inquirer.Checkbox('tasks',
                         message="Select tasks to execute (Space to select, Enter to confirm)",
                         choices=[task[0] for task in tasks])
    ]
    
    answers = inquirer.prompt(questions)
    selected_tasks = answers['tasks']
    
    if not selected_tasks:
        print("No tasks selected.")
        return
    
    setup_logging()
    
    for task_name in selected_tasks:
        task_id = task_name.split('.')[0]
        print(f"\nExecuting {task_name}...")
        
        if task_id == "1":
            download_spy()
            
        elif task_id == "2":
            try:
                filter_stocks(
                    input_file='Data.csv',
                    output_file='filtered_stock.csv',
                    min_trading_days=1000,
                    start_date='2001-01-01',
                    end_date='2024-10-01',
                )
                print("Outlier filtering completed")
            except Exception as e:
                print(f"Outlier filtering failed: {str(e)}")
            
        elif task_id == "3":
            create_return_df()
        elif task_id == "4":
            create_SP500_return_df()
    
    print("\nAll selected tasks completed.")

if __name__ == "__main__":
    main()