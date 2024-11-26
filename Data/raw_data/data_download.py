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
from scipy import stats

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

def main():
    """Main execution function for data download"""
    tasks = [
        ("1. Extract symbols (create full_ticker.csv)", "extract_symbols"),
        ("2. Download and filter stocks (create filtered_ticker.csv)", "filter_stocks"),
        ("3. Check and download S&P 500 stocks", "check_sp500"),
        ("4. Create final dataset (Data.csv)", "create_dataset")
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
        
    downloader = StockDataDownloader()
    
    for task_name in selected_tasks:
        task_id = task_name.split('.')[0]
        print(f"\nExecuting {task_name}...")
        
        if task_id == "1":
            if os.path.exists('full_ticker.csv'):
                symbols = pd.read_csv('full_ticker.csv')
                print(f"Already downloaded symbols: {len(symbols)}")
            else:
                symbols = downloader.get_exchange_symbols()
                symbols.to_csv('full_ticker.csv', index=False)
                
        elif task_id == "2":
            if os.path.exists('filtered_ticker.csv'):
                filtered_symbols = pd.read_csv('filtered_ticker.csv')
                print(f"Already downloaded symbols: {len(filtered_symbols)}")
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
                print(f"New symbols added: {len(filtered_symbols)}")
    
    print("\nAll selected tasks completed.")

if __name__ == "__main__":
    main()