import pandas as pd
import os

def filter_stocks(
    input_file='all_data.csv', 
    output_file='filtered_stock_1985.csv', 
    min_trading_days=1000,
    start_date='1985-01-01', 
    end_date='2024-09-01'
):
    """
    Trims stock data to the specified date range and optionally filters stocks 
    with at least a given number of trading days. All stocks remain, but only 
    data within the date range is kept.

    Args:
        input_file (str): Name of the input CSV file.
        output_file (str): Name of the output CSV file.
        min_trading_days (int): Minimum required trading days (for filtering only).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        None
    """
    # Read the input CSV file and parse dates
    df = pd.read_csv(input_file, parse_dates=['date'])

    # Trim data to the specified date range
    trimmed_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    # Calculate the number of trading days per stock in the trimmed data
    trading_days = trimmed_df.groupby('PERMNO')['date'].count()

    # Filter stocks with at least the minimum trading days (optional filtering)
    valid_permnos = trading_days[trading_days >= min_trading_days].index

    # Select only valid stocks based on trading day condition (optional)
    filtered_df = trimmed_df[trimmed_df['PERMNO'].isin(valid_permnos)]

    # Save the trimmed (and optionally filtered) data to a new CSV file
    filtered_df.to_csv(output_file, index=False)

    print(f"Original number of stocks: {df['PERMNO'].nunique()}")
    print(f"Number of stocks with at least {min_trading_days} trading days: {len(valid_permnos)}")
    print(f"Trimmed data saved to {output_file}.")

if __name__ == "__main__":
    filter_stocks()
