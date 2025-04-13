# import pandas as pd
# import numpy as np

# def run_backtest(df, signals, initial_cash=10000):
#     df, signals = df.align(signals, join='inner', axis=0)  # This ensures alignment
#     cash = initial_cash
#     holdings = 0
#     portfolio_values = []

#     for i in range(len(df)):
#         signal = signals.iloc[i]
#         price = df['Close'].iloc[i]

#         if signal == 1 and holdings == 0:
#             holdings = cash / price
#             cash = 0
#         elif signal == -1 and holdings > 0:
#             cash = holdings * price
#             holdings = 0

#         portfolio_values.append(cash + holdings * price)

#     results = pd.DataFrame({
#         'PortfolioValue': portfolio_values,
#         'Close': df['Close']
#     }, index=df.index)

#     return results


import pandas as pd
import numpy as np

def run_backtest(df, signals, initial_cash=10000):
    df, signals = df.align(signals, join='inner', axis=0)  # This ensures alignment
    cash = initial_cash
    holdings = 0
    portfolio_values = []

    for i in range(len(df)):
        signal = signals.iloc[i]
        price = df['Close'].iloc[i]

        if signal == 1 and holdings == 0:
            holdings = cash / price
            cash = 0
        elif signal == -1 and holdings > 0:
            cash = holdings * price
            holdings = 0

        portfolio_values.append(cash + holdings * price)

    results = pd.DataFrame({
        'PortfolioValue': portfolio_values,
        'Close': df['Close']
    }, index=df.index)

    # Calculate performance metrics
    results['Returns'] = results['PortfolioValue'].pct_change()
    sharpe_ratio = results['Returns'].mean() / results['Returns'].std() * np.sqrt(252)
    results.attrs['sharpe_ratio'] = sharpe_ratio

    return results
