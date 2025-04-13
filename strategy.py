# import pandas as pd
# from sklearn.linear_model import LogisticRegression

# def compute_rsi(close, window=14):
#     delta = close.diff()
#     gain = delta.where(delta > 0, 0.0)
#     loss = -delta.where(delta < 0, 0.0)
#     avg_gain = gain.rolling(window=window).mean()
#     avg_loss = loss.rolling(window=window).mean()
#     rs = avg_gain / avg_loss
#     return 100 - (100 / (1 + rs))

# def rule_based_strategy(df):
#     df = df.copy()

#     if 'Close' not in df.columns:
#         raise ValueError("Missing 'Close' column in the input DataFrame")

#     df['SMA20'] = df['Close'].rolling(window=20).mean()
#     df['RSI14'] = compute_rsi(df['Close'])

#     # Debugging step: Inspect DataFrame columns here
#     print("Current DataFrame columns:", df.columns.tolist())

#     # Ensure columns exist explicitly before dropping NaNs
#     required_cols = ['Close', 'SMA20', 'RSI14']
#     for col in required_cols:
#         if col not in df.columns:
#             raise KeyError(f"Column '{col}' is missing before dropping NaNs.")

#     df.dropna(subset=required_cols, inplace=True)

#     signals = pd.Series(0, index=df.index)
#     signals[(df['RSI14'] < 30) & (df['Close'] < df['SMA20'])] = 1
#     signals[(df['RSI14'] > 70) & (df['Close'] > df['SMA20'])] = -1

#     return signals



# def ml_based_strategy(df):
#     df = df.copy()
#     df['Future_Return'] = df['Close'].shift(-1) > df['Close']
#     df.dropna(inplace=True)

#     X = df[['RSI14', 'MACD', 'MACD_signal', 'SMA20', 'SMA50']]
#     y = df['Future_Return'].astype(int)

#     split = int(len(df)*0.7)
#     X_train, X_test = X[:split], X[split:]
#     y_train, y_test = y[:split], y[split:]

#     model = LogisticRegression(max_iter=500)
#     model.fit(X_train, y_train)

#     prob = model.predict_proba(X)[:,1]
#     signals = pd.Series(0, index=df.index)
#     signals[prob > 0.6] = 1
#     signals[prob < 0.4] = -1
#     return signals


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def compute_rsi(close, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def rule_based_strategy(df):
    df = df.copy()

    if 'Close' not in df.columns:
        raise ValueError("Missing 'Close' column in the input DataFrame")

    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['RSI14'] = compute_rsi(df['Close'])

    print("Current DataFrame columns:", df.columns.tolist())

    required_cols = ['Close', 'SMA20', 'RSI14']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' is missing before dropping NaNs.")

    df.dropna(subset=required_cols, inplace=True)

    signals = pd.Series(0, index=df.index)
    signals[(df['RSI14'] < 30) & (df['Close'] < df['SMA20'])] = 1
    signals[(df['RSI14'] > 70) & (df['Close'] > df['SMA20'])] = -1

    return signals


def ml_based_strategy(df):
    """
    Implements an ML-based trading strategy using Logistic Regression.
    
    The function:
      1. Calculates the target variable ('Future_Return') as 1 if the next close is greater than the current close (and 0 otherwise).
      2. Extracts technical indicators (RSI14, MACD, MACD_signal, SMA20, SMA50) as features.
      3. Splits the data into training and testing sets.
      4. Trains Logistic Regression to predict the target variable.
      5. Generates a probability for a positive return.
      6. Assigns a Buy signal (1) if the probability is greater than 0.6, a Sell signal (-1) if it is less than 0.4, and Hold (0) in between.
    
    Returns:
        signals (pd.Series): Trading signals indexed as 1 (buy), -1 (sell), or 0 (hold).
    """
    df = df.copy()
    
    # Create target variable: 1 if next period close > current period close, else 0.
    df['Future_Return'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    
    # Use technical indicators as features.
    # These should have already been computed by add_indicators or similar preprocessing.
    X = df[['RSI14', 'MACD', 'MACD_signal', 'SMA20', 'SMA50']]
    y = df['Future_Return']
    
    # Split the data (70% train, 30% test)
    split = int(len(df) * 0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    # Train a Logistic Regression model.
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    
    # Evaluate the model; store the accuracy in the DataFrame attributes.
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    df.attrs['ml_accuracy'] = accuracy
    
    # Get the probability of the positive class (i.e., expecting the price to increase).
    prob = model.predict_proba(X)[:, 1]
    
    # Initialize signals as hold (0) for all time steps.
    signals = pd.Series(0, index=df.index)
    
    # If probability > 0.6, signal Buy; if probability < 0.4, signal Sell.
    signals[prob > 0.6] = 1
    signals[prob < 0.4] = -1
    
    return signals
