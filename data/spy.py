# SPY 다운로드 코드
import yfinance as yf

spy = yf.download('SPY', start='1985-01-01', end='2024-09-01')
spy.to_csv('data/spy.csv')