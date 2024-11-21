import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
from pathlib import Path
import os
import inquirer
from datetime import datetime

class StockDataDownloader:
    """주식 데이터 다운로드 및 처리를 위한 클래스"""
    
    def __init__(self):
        """초기화 및 기본 설정"""
        self.setup_logging()
        self.create_directories()
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            filename='download.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def create_directories(self):
        """필요한 디렉토리 생성"""
        Path('./stock_data').mkdir(exist_ok=True)
    
    def get_exchange_symbols(self):
        """
        나스닥과 뉴욕거래소의 종목 정보 추출 및 이미 다운로드된 종목 제외
        
        Returns:
            DataFrame: 다운로드가 필요한 종목 정보
        """
        # 거래소별 종목 리스트 다운로드
        nasdaq = fdr.StockListing('NASDAQ')
        nyse = fdr.StockListing('NYSE')
        
        # 필요한 컬럼만 선택하고 거래소 정보 추가
        nasdaq['Exchange'] = 'NASDAQ'
        nyse['Exchange'] = 'NYSE'
        
        # 데이터프레임 통합
        symbols = pd.concat([nasdaq, nyse])
        symbols = symbols[['Symbol', 'Name', 'IndustryCode', 'Industry', 'Exchange']]
        
        # 이미 다운로드된 종목 제외
        if os.path.exists('symbol_permno.csv'):
            downloaded = pd.read_csv('symbol_permno.csv')
            symbols = symbols[~symbols['Symbol'].isin(downloaded['Symbol'])]
            logging.info(f"Excluding {len(downloaded)} already downloaded symbols")
        
        return symbols
    
    def download_and_filter_stocks(self, symbols):
        """
        종목 데이터 다운로드 및 필터링 (5000일 이상)
        
        Args:
            symbols (DataFrame): 다운로드할 종목 정보
            
        Returns:
            DataFrame: 필터링된 종목 정보
        """
        if len(symbols) == 0:
            logging.info("No new symbols to download")
            return pd.DataFrame()
        
        filtered_symbols = []
        
        def process_symbol(row):
            try:
                data = fdr.DataReader(row['Symbol'])
                if len(data) >= 5000:
                    # 조정 계수 계산
                    adj_factor = data['Adj Close'] / data['Close']
                    
                    # 가격 데이터 조정
                    data['Open'] = data['Open'] * adj_factor
                    data['High'] = data['High'] * adj_factor
                    data['Low'] = data['Low'] * adj_factor
                    data['Close'] = data['Adj Close']
                    
                    # 결측치 처리
                    data = data.ffill()
                    
                    # 심볼 및 날짜 정보 추가
                    data['Symbol'] = row['Symbol']
                    data.index.name = 'date'
                    
                    # 파일 저장
                    data.to_csv(f'./stock_data/{row["Symbol"]}.csv')
                    logging.info(f"Downloaded and processed {row['Symbol']}")
                    return row
                else:
                    logging.info(f"Skipped {row['Symbol']} (insufficient data: {len(data)} days)")
                    return None
            except Exception as e:
                logging.error(f"Error processing {row['Symbol']}: {e}")
                return None
        
        # 멀티스레딩으로 다운로드 및 필터링
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_symbol, row) for _, row in symbols.iterrows()]
            
            for future in tqdm(as_completed(futures), total=len(symbols), 
                             desc="Downloading and filtering"):
                result = future.result()
                if result is not None:
                    filtered_symbols.append(result)
        
        filtered_df = pd.DataFrame(filtered_symbols)
        
        # 기존 symbol_permno.csv에 새로운 종목 추가
        if os.path.exists('symbol_permno.csv'):
            existing_symbols = pd.read_csv('symbol_permno.csv')
            next_permno = existing_symbols['PERMNO'].max() + 1
            
            new_symbols = pd.DataFrame({
                'Symbol': filtered_df['Symbol'],
                'PERMNO': range(next_permno, next_permno + len(filtered_df))
            })
            
            updated_symbols = pd.concat([existing_symbols, new_symbols])
            updated_symbols.to_csv('symbol_permno.csv', index=False)
        else:
            symbol_permno = pd.DataFrame({
                'Symbol': filtered_df['Symbol'],
                'PERMNO': range(1, len(filtered_df) + 1)
            })
            symbol_permno.to_csv('symbol_permno.csv', index=False)
        
        logging.info(f"Added {len(filtered_df)} new symbols")
        return filtered_df
    
    def create_final_dataset(self, symbols):
        """
        최종 데이터셋 생성
        
        Args:
            symbols (DataFrame): 필터링된 종목 정보
        """
        dfs = []
        
        for file in tqdm(Path('./stock_data').glob('*.csv'), desc="Merging data"):
            try:
                df = pd.read_csv(file)
                df['date'] = pd.to_datetime(df['date'])
                
                # 수익률 계산
                df.sort_values('date', inplace=True)
                df['Ret'] = df['Close'].pct_change()
                
                # 결측치 제거
                df.dropna(subset=['Ret'], inplace=True)
                
                # 종목 정보 추가
                symbol = file.stem
                symbol_info = symbols[symbols['Symbol'] == symbol].iloc[0]
                df['Exchange'] = symbol_info['Exchange']
                df['IndustryCode'] = symbol_info['IndustryCode']
                
                dfs.append(df)
            except Exception as e:
                logging.error(f"Error processing {file}: {e}")
        
        if dfs:
            # 데이터 통합
            final_data = pd.concat(dfs, ignore_index=True)
            
            # PERMNO 생성
            symbol_permno = pd.DataFrame({
                'Symbol': symbols['Symbol'].unique(),
                'PERMNO': range(1, len(symbols['Symbol'].unique()) + 1)
            })
            
            # Symbol-PERMNO 매핑 저장
            symbol_permno.to_csv('symbol_permno.csv', index=False)
            
            # 최종 데이터셋 생성
            final_data = final_data.merge(symbol_permno, on='Symbol')
            
            # 날짜 형식 변경
            final_data['date'] = final_data['date'].dt.strftime('%Y%m%d')
            
            # 컬럼 이름 변경
            columns = {
                'date': 'date',
                'PERMNO': 'PERMNO',
                'Low': 'BIDLO',
                'High': 'ASKHI',
                'Close': 'PRC',
                'Volume': 'VOL',
                'Open': 'OPENPRC',
                'Ret': 'RET'
            }
            
            final_data = final_data.rename(columns=columns)
            final_data['SHROUT'] = np.nan  # Shares Outstanding 정보 없음
            
            # 최종 컬럼 선택 및 정렬
            final_columns = ['date', 'PERMNO', 'BIDLO', 'ASKHI', 'PRC', 
                            'VOL', 'SHROUT', 'OPENPRC', 'RET']
            
            final_data = final_data[final_columns]
            final_data = final_data.round(3)
            final_data.sort_values(['PERMNO', 'date'], inplace=True)
            
            # 데이터 저장
            final_data.to_csv('Data.csv', index=False)
            
            logging.info(f"Final dataset created with {len(symbol_permno)} symbols")
    
    def check_sp500(self, symbols):
        """
        S&P 500 종목 확인 및 누락된 종목 다운로드
        오류 발생 시 sp500_ticker.csv에 오류 유형 기록
        
        Parameters
        ----------
        symbols : pandas.DataFrame
            종목 정보가 포함된 데이터프레임
            
        Returns
        -------
        None
        """
        try:
            # S&P 500 종목 리스트 로드
            sp500 = pd.read_csv('sp500_20180101.csv')
            sp500_symbols = set(sp500['Symbol'])
            
            # 입력받은 symbols에서 현재 보유 종목 확인
            downloaded_symbols = set(symbols['Symbol'])
            
            # 이전 오류 기록 확인
            error_records = pd.DataFrame()
            if os.path.exists('sp500_ticker.csv'):
                error_records = pd.read_csv('sp500_ticker.csv')
                error_symbols = set(error_records['Symbol'])
            else:
                error_symbols = set()
                
            # 누락된 S&P 500 종목 확인 (이전 오류 제외)
            missing_sp500 = sp500_symbols - downloaded_symbols - error_symbols
                
            if missing_sp500:
                logging.info(f"다운로드 시도할 S&P 500 종목 수: {len(missing_sp500)}")
                
                # 누락된 종목 다운로드 시도 및 오류 기록
                error_list = []
                for symbol in tqdm(missing_sp500, desc="Downloading missing S&P 500 stocks", ncols=100, position=0, leave=True):
                    try:
                        missing_df = pd.DataFrame({'Symbol': [symbol]})
                        self.download_and_filter_stocks(missing_df)
                    except Exception as e:
                        error_list.append({
                            'Symbol': symbol,
                            'Error_Type': type(e).__name__,
                            'Error_Message': str(e),
                            'Timestamp': pd.Timestamp.now()
                        })
                
                # 오류 기록 저장
                if error_list:
                    new_errors = pd.DataFrame(error_list)
                    error_records = pd.concat([error_records, new_errors], ignore_index=True)
                    error_records.to_csv('sp500_ticker.csv', index=False)
                    logging.info(f"오류 발생 종목 수: {len(error_list)}")
            else:
                logging.info("다운로드가 필요한 S&P 500 종목이 없습니다.")
                
        except Exception as e:
            logging.error(f"S&P 500 종목 확인 중 오류 발생: {e}")
    
    def download_spy(self):
        """SPY ETF 데이터 다운로드"""
        try:
            spy_data = fdr.DataReader('S&P500')
            spy_data.index.name = 'Date'
            spy_data.to_csv('snp500_index.csv')
            logging.info("SPY 데이터 다운로드 완료")
        except Exception as e:
            logging.error(f"SPY 데이터 다운로드 실패: {e}")

def create_return_df():
    """
    SP500 2018년 구성종목들의 수익률 데이터프레임을 생성합니다.
    
    Returns:
        pd.DataFrame: 인덱스는 날짜, 컬럼은 종목코드인 수익률 데이터프레임
    """
    import pandas as pd
    from pathlib import Path
    
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
    return_df.to_csv(data_dir / 'return_df.csv')
    print(f"수익률 데이터 생성 완료: {return_df.shape}")
    print(f"기간: {return_df.index.min()} ~ {return_df.index.max()}")
    print(f"종목 수: {len(return_df.columns)}")
    
    return return_df

def setup_logging(log_dir='logs'):
    """로깅 설정을 초기화합니다."""
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
    
    logging.info("로깅 시작")
    return log_file

def detect_abnormal_stocks(returns, prices, price_jumps_threshold=0.5, recovery_window=5, price_gap_threshold=10):
    """주가 데이터의 이상치를 탐지하는 향상된 함수입니다."""
    stocks_to_remove = set()
    
    # 전체 기간 수익률이 -100% 이하인 종목 탐지
    for stock in prices.columns:
        price_series = prices[stock].dropna()
        if len(price_series) < 2:
            continue
            
        total_return = (price_series.iloc[-1] / price_series.iloc[0]) - 1
        if total_return <= -1.0:
            stocks_to_remove.add(stock)
            logging.info(f"PERMNO {stock}: 전체 기간 수익률 {total_return:.1%}로 제거 대상")
    
    # 급격한 가격 하락 후 빠른 회복 패턴 탐지
    for stock in prices.columns:
        price_series = prices[stock].dropna()
        if len(price_series) < recovery_window:
            continue
            
        for i in range(len(price_series) - recovery_window):
            window = price_series.iloc[i:i+recovery_window]
            initial_price = window.iloc[0]
            min_price = window.min()
            final_price = window.iloc[-1]
            
            if (min_price < initial_price * 0.1 and final_price > initial_price * 0.8):
                stocks_to_remove.add(stock)
                break
    
    # 연속된 거래일 간의 비정상적인 가격 변동 탐지
    for stock in prices.columns:
        price_series = prices[stock].dropna()
        daily_changes = price_series.pct_change().abs()
        
        if (daily_changes > price_gap_threshold).any():
            stocks_to_remove.add(stock)
    
    # 이동평균을 이용한 이상치 탐지
    window_size = 20
    for stock in prices.columns:
        price_series = prices[stock].dropna()
        if len(price_series) < window_size:
            continue
            
        rolling_mean = price_series.rolling(window=window_size).mean()
        rolling_std = price_series.rolling(window=window_size).std()
        
        z_scores = (price_series - rolling_mean) / rolling_std
        if (abs(z_scores) > 5).any():
            stocks_to_remove.add(stock)
    
    logging.info(f"이상치 탐지 결과:")
    logging.info(f"- 전체 기간 수익률 -100% 이하 종목 수: {len([s for s in stocks_to_remove if (prices[s].iloc[-1] / prices[s].iloc[0] - 1) <= -1.0])}개")
    logging.info(f"- 총 제거 대상 종목 수: {len(stocks_to_remove)}개")
    
    return list(stocks_to_remove)

def filter_stocks(
    input_file, 
    output_file, 
    min_trading_days=1000,
    start_date='2001-01-01', 
    end_date='2024-08-01',
    price_jumps_threshold=0.5,
    recovery_window=5,
    price_gap_threshold=10
):
    """이상치 데이터를 가진 종목을 제거하여 주식 데이터를 필터링합니다."""
    try:
        logging.info(f"데이터 필터링 시작: {input_file}")
        logging.info(f"파라미터 - 최소거래일: {min_trading_days}, 시작일: {start_date}, 종료일: {end_date}")
    
        df = pd.read_csv(input_file, parse_dates=['date'])
        initial_count = len(df)
        logging.info(f"초기 데이터 수: {initial_count:,}행")
    
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        date_filtered_count = len(df)
        logging.info(f"날짜 필터링 후 데이터 수: {date_filtered_count:,}행 (제거됨: {initial_count - date_filtered_count:,}행)")
    
        trading_days = df.groupby('PERMNO')['date'].count()
        valid_permnos = trading_days[trading_days >= min_trading_days].index
        df = df[df['PERMNO'].isin(valid_permnos)]
        
        logging.info(f"거래일수 기준 필터링 후 주식 수: {len(valid_permnos):,}개")
    
        prices_df = df.pivot(index='date', columns='PERMNO', values='PRC').abs()
        returns_df = df.pivot(index='date', columns='PERMNO', values='RET')
        
        logging.info(f"피벗 데이터 형태 - 가격: {prices_df.shape}, 수익률: {returns_df.shape}")
    
        stocks_to_remove = detect_abnormal_stocks(
            returns=returns_df,
            prices=prices_df,
            price_jumps_threshold=price_jumps_threshold,
            recovery_window=recovery_window,
            price_gap_threshold=price_gap_threshold
        )
        
        df_cleaned = df[~df['PERMNO'].isin(stocks_to_remove)]
        final_count = len(df_cleaned)
        logging.info(f"이상치 종목 제거 후 데이터 수: {final_count:,}행 (제거됨: {initial_count - final_count:,}행)")
    
        df_cleaned.to_csv(output_file, index=False)
        logging.info(f"최종 데이터 저장 완료: {output_file}")
    
        return df_cleaned
    
    except Exception as e:
        logging.error(f"에러 발생: {str(e)}", exc_info=True)
        raise

def main():
    """메인 실행 함수"""
    tasks = [
        ("1. 종목 정보 추출 (full_ticker.csv 생성)", "extract_symbols"),
        ("2. 데이터 다운로드 및 필터링 (filtered_ticker.csv 생성)", "filter_stocks"),
        ("3. S&P 500 종목 확인 및 다운로드", "check_sp500"),
        ("4. 최종 데이터셋 생성 (Data.csv)", "create_dataset"),
        ("5. SPY ETF 데이터 다운로드", "download_spy"),
        ("6. 이상치 데이터 필터링 (filtered_stock.csv)", "filter_abnormal"),
        ("7. 수익률 데이터프레임 생성 (return_df.csv)", "create_return")
    ]
    
    questions = [
        inquirer.Checkbox('tasks',
                         message="실행할 작업을 선택하세요 (스페이스바로 선택, 엔터로 확인)",
                         choices=[task[0] for task in tasks])
    ]
    
    answers = inquirer.prompt(questions)
    selected_tasks = answers['tasks']
    
    if not selected_tasks:
        print("작업이 선택되지 않았습니다.")
        return
        
    downloader = StockDataDownloader()
    
    for task_name in selected_tasks:
        task_id = task_name.split('.')[0]
        print(f"\n{task_name} 실행 중...")
        
        if task_id == "1":
            if os.path.exists('full_ticker.csv'):
                symbols = pd.read_csv('full_ticker.csv')
                print(f"이미 다운로드된 종목 수: {len(symbols)}")
            else:
                symbols = downloader.get_exchange_symbols()
                symbols.to_csv('full_ticker.csv', index=False)
                
        elif task_id == "2":
            if os.path.exists('filtered_ticker.csv'):
                filtered_symbols = pd.read_csv('filtered_ticker.csv')
                print(f"이미 다운로드된 종목 수: {len(filtered_symbols)}")
            else:
                symbols = pd.read_csv('full_ticker.csv')
                filtered_symbols = downloader.download_and_filter_stocks(symbols)
                
        elif task_id == "3":
            filtered_symbols = pd.read_csv('filtered_ticker.csv')
            downloader.check_sp500(filtered_symbols)
            
        elif task_id == "4":
            filtered_symbols = pd.read_csv('filtered_ticker.csv')
            if len(filtered_symbols) > 0:
                downloader.create_final_dataset(filtered_symbols)
                print(f"새로 추가된 종목 수: {len(filtered_symbols)}")
                
        elif task_id == "5":
            downloader.download_spy()
            
        elif task_id == "6":
            try:
                filter_stocks(
                    input_file='Data.csv',
                    output_file='filtered_stock.csv',
                    min_trading_days=1000,
                    start_date='2001-01-01',
                    end_date='2024-10-01',
                    price_jumps_threshold=0.75,
                    recovery_window=5,
                    price_gap_threshold=10
                )
                print("이상치 필터링 완료")
            except Exception as e:
                print(f"이상치 필터링 실패: {str(e)}")
            
        elif task_id == "7":
            create_return_df()
    
    print("\n선택한 모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()
    