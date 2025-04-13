# indicators.py
import pandas as pd
import numpy as np

def fractional_difference_weights(d, thresh=1e-5, max_len=None):
    """
    Compute weights for fractional differencing with order d.
    
    Parameters:
        d (float): The fractional differencing order.
        thresh (float): Threshold to stop computing further weights.
        max_len (int, optional): Maximum number of weights allowed.
        
    Returns:
        np.array: Array of weights in reversed order.
    """
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
        k += 1
        # Prevent generating more weights than there are data points
        if max_len is not None and k >= max_len:
            break
    return np.array(w[::-1])

def fractional_difference(series, d, thresh=1e-5):
    """
    Apply fractional differencing to a time series.
    
    Parameters:
        series (pd.Series): The original time series (e.g., Close prices).
        d (float): The differencing order (e.g., 0.5).
        thresh (float): Weight threshold for truncation.
        
    Returns:
        pd.Series: The fractionally differenced series with NaN values
                   for the initial period.
    """
    # Ensure we do not generate more weights than the series has data
    weights = fractional_difference_weights(d, thresh=thresh, max_len=len(series))
    width = len(weights)
    diff_series = []
    
    # Loop over the series starting from the point where a full window is available
    for i in range(width - 1, len(series)):
        window = series.iloc[i - width + 1 : i + 1]
        diff_value = np.dot(weights, window)
        diff_series.append(diff_value)
    
    # Create a new series with NaNs for the initial period where a full window wasn't available
    result = pd.Series([np.nan] * (width - 1) + diff_series, index=series.index)
    return result

def add_indicators(df, sma_short=20, sma_long=50, rsi_window=14, frac_order=0.5):
    """
    Add technical indicators and a fractionally differenced Close price to the DataFrame.
    
    The function adds the following columns:
      - SMA20: 20-period simple moving average of Close.
      - SMA50: 50-period simple moving average of Close.
      - RSI14: 14-period Relative Strength Index.
      - MACD: Difference between the 12-period and 26-period exponential moving averages.
      - MACD_signal: 9-period exponential moving average of MACD.
      - FracDiff_Close: Fractionally differenced Close price using the specified order.
    
    It resets the index before performing the calculations and drops any rows that 
    result in NaN values.
    
    Parameters:
        df (pd.DataFrame): DataFrame with at least a 'Close' column (and optionally 'High', 'Low', 'Open', 'Volume').
        sma_short (int): Window for the short-term simple moving average.
        sma_long (int): Window for the long-term simple moving average.
        rsi_window (int): Window for the RSI calculation.
        frac_order (float): Order for fractional differencing applied to 'Close'.
        
    Returns:
        pd.DataFrame: Updated DataFrame with all indicator columns added.
    """
    # Reset the index to ensure a simple 0..N-1 RangeIndex
    df = df.copy().reset_index(drop=True)
    
    # Compute Simple Moving Averages
    df['SMA20'] = df['Close'].rolling(window=sma_short).mean()
    df['SMA50'] = df['Close'].rolling(window=sma_long).mean()
    
    # RSI calculation
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=rsi_window).mean()
    loss = -delta.clip(upper=0).rolling(window=rsi_window).mean()
    rs = gain / loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    
    # MACD and MACD Signal
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    
    # Fractionally Differenced Close Price
    try:
        df['FracDiff_Close'] = fractional_difference(df['Close'], frac_order)
    except Exception as e:
        raise ValueError("Error computing fractional differenced series: " + str(e))
    
    # Drop rows with any NaN values and reset the index for consistency
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df
