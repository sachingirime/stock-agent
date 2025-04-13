# # data.py
# import yfinance as yf
# import pandas as pd

# def fetch_historical_data(symbol, start, end, interval="1d"):
#     df = yf.download(symbol, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
#     df.columns = df.columns.get_level_values(0)
#     return df.dropna().sort_index()

# # data.py
# def fetch_realtime_data(symbol, period="1d", interval="1m"):
#     df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
#     df.columns = df.columns.get_level_values(0)
#     df = df.dropna().sort_index()
    
#     # Debugging: clearly print out latest timestamp fetched
#     print("Last timestamp fetched from yfinance:", df.index[-1])
#     return df


# import yfinance as yf
# import pandas as pd

# def fetch_historical_data(symbol, start, end, interval="1d"):
#     df = yf.download(symbol, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = df.columns.get_level_values(0)
#     if 'Close' not in df.columns:
#         raise ValueError("The fetched DataFrame does not contain a 'Close' column.")
#     return df.dropna().sort_index()

# def fetch_realtime_data(symbol, period="1d", interval="1m"):
#     df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
#     if isinstance(df.columns, pd.MultiIndex):
#         df.columns = df.columns.get_level_values(0)
#     if 'Close' not in df.columns:
#         raise ValueError("The fetched DataFrame does not contain a 'Close' column.")
#     df = df.dropna().sort_index()
#     print(f"[{symbol}] Last data point: {df.index[-1]} | Total rows: {len(df)}")
#     return df

# data.py
import yfinance as yf
import pandas as pd

def fetch_historical_data(symbol, start, end, interval="1d"):
    """
    Fetch daily (or sub-daily) data by specifying start and end dates.
    Note: For intervals < 1d, Yahoo Finance often enforces strict limits on date ranges.
    """
    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if 'Close' not in df.columns:
        raise ValueError("The fetched DataFrame does not contain a 'Close' column.")
    return df.dropna().sort_index()

def fetch_intraday_data(symbol, period="5d", interval="5m"):
    """
    Fetch intraday historical data for the specified period and interval.
    Examples:
      period='1d', '5d', '1mo', '3mo', '6mo', '1y'
      interval='1m', '5m', '15m', '30m', '60m'
    
    For example, period='30d' and interval='15m' might work.
    But 1m data typically only goes up to 7 days.
    """
    df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if 'Close' not in df.columns:
        raise ValueError("No 'Close' column found. Check symbol or data availability.")
    
    df = df.dropna().sort_index()
    print(f"[{symbol}] Last data point: {df.index[-1]} | Total rows: {len(df)}")
    return df
