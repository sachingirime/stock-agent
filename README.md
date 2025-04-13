USAGES: \n
1. Install all  the dependencies
2. In command terminal --> streamlit run main.py \n

🌍 Project Overview
The stock market is complex, dynamic, and fast-paced. As someone passionate about both finance and artificial intelligence, I decided to build an intelligent stock trading assistant powered by machine learning and reinforcement learning (RL). The result? A live, interactive dashboard that simulates stock trading strategies and visualizes insights in real time. You can also train your own model with whatever parameters you provide and do the inference.

📊 Live demo: predictpp.streamlit.app

✨ Features at a Glance
✅ Real-time stock data from Yahoo Finance (via yfinance)
⚖️ Rule-based and ML-based strategy comparison
🧠 RL agent (PPO) trained to make intelligent buy/sell/hold decisions
📊 Plotly graphs with zoom/pan and buy/sell signal overlays
⌚ Auto-refresh and live chart training metrics
⭐ Prediction accuracy metrics
⚡ Technology Stack
Python
Streamlit
Plotly
Stable Baselines3 (PPO RL algorithm)
scikit-learn (ML classifier)
yfinance (market data)
🎓 Key Learning Objectives
Explore and compare trading strategies: Rule-based, supervised ML, and RL
Visualize trade decisions and portfolio performance interactively
Build and deploy a data-driven AI system on the web
⚖️ Trading Strategies Implemented
1. Rule-Based Strategy
Combines two popular technical indicators:

SMA-20: Simple Moving Average
RSI-14: Relative Strength Index
Buy Signal: RSI < 30 and price below SMA-20
Sell Signal: RSI > 70 and price above SMA-20

2. Machine Learning Strategy
Uses a logistic regression classifier trained on:

RSI, MACD, SMA indicators
Target: whether the price will increase in the next timestep
Outputs probabilistic confidence to classify actions as Buy / Hold / Sell.

3. Reinforcement Learning Agent
Built using Proximal Policy Optimization (PPO):

Custom OpenAI Gym environment for trading
Observations: Indicators + position state
Rewards: Profit/loss at each step
Actions: {0: Sell, 1: Hold, 2: Buy}
🛠️ Feature Engineering with Fractional Order Indicators
To strengthen signal quality, especially under volatile conditions, we integrated a custom feature engineering approach using Fractional Order Differencing. This method helps retain long-term memory in time series data while stabilizing variance.

Fractional derivatives enhance signal smoothness for indicators like RSI and MACD.
This provides a more nuanced view of trends and reversals.
Combined with classical features (SMA, MACD, etc.), this enriches the observation space for the RL agent.
This additional preprocessing has proven useful in training more stable and generalizable policies.

🌐 Visual Insights
The dashboard uses Plotly for:

Buy/Sell markers on candlestick-like price plots
Live training metrics: loss, policy gradient, KL divergence
Zoom and pan for deep-dive exploration
⚖️ Model Evaluation
To evaluate RL model accuracy:

Compare predicted action direction to actual next-price movement
Compute overall prediction accuracy and display as a metric
🚀 Deployment
Deployment was made easy with Streamlit Cloud. I simply pushed the code to GitHub and connected the repo.

Live dashboard: predictpp.streamlit.app

🚧 Future Enhancements
Add LSTM-based time series forecasting
Portfolio diversification
Integration with Alpaca or Robinhood for real trading
💡 Final Thoughts
This project is a step toward democratizing financial insights using modern AI techniques. Whether you’re a data scientist, finance enthusiast, or an investor, this tool provides a meaningful interface to explore how algorithms can augment decision-making in volatile markets.

The beauty lies in the blend: human intuition + algorithmic automation.

📖 GitHub — CODE HERE
github.com/sachingirime/stock-agent

🌐 Live App
predictpp.streamlit.app
