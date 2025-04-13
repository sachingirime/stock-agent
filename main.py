# # # main.py
# # import streamlit as st
# # from data import fetch_historical_data, fetch_realtime_data
# # from indicators import add_indicators
# # from strategy import rule_based_strategy, ml_based_strategy
# # from backtest import run_backtest
# # from visualize import run_dashboard
# # import pandas as pd
# # from streamlit_autorefresh import st_autorefresh


# # def main():
# #         # Auto-refresh every 5 seconds during market hours, change as needed
# #     if st.sidebar.checkbox("🔄 Auto-refresh (5-seconds intervals)"):
# #         st_autorefresh(interval= 5 * 1000)  # currently set to 60 seconds
 
# #     st.sidebar.title("⚙️ Trading Pipeline Settings")
    
# #     symbol = st.sidebar.text_input("Symbol:", "TSLA")

# #     mode = st.sidebar.radio("Select Data Mode:", ["Historical Daily", "Real-Time Intraday"])

# #     if mode == "Historical Daily":
# #         start = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
# #         end = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-01-01"))
# #         df = fetch_historical_data(symbol, start, end, interval="1d")
# #     else:
# #         period = st.sidebar.selectbox("Intraday Period:", ["1d", "5d"])
# #         interval = st.sidebar.selectbox("Intraday Interval:", ["1m", "5m", "15m"])
# #         df = fetch_realtime_data(symbol, period=period, interval=interval)

    

# #     if df.empty:
# #         print("❌ No data returned from Yahoo Finance. Try a longer period or different interval.")
# #         return

# #     df = add_indicators(df)

# #     strategy_choice = st.sidebar.selectbox("Choose Strategy", ["Rule-Based", "ML-Based"])

# #     if strategy_choice == "Rule-Based":
# #         signals = rule_based_strategy(df)
# #     else:
# #         signals = ml_based_strategy(df)

# #     results = run_backtest(df, signals)

# #     run_dashboard(results, signals, df)

# # if __name__ == "__main__":
# #     main()


# # # main.py
# # import streamlit as st
# # import pandas as pd
# # from data import fetch_historical_data
# # from indicators import add_indicators
# # from model import train_rl_agent, load_rl_agent, predict_action
# # from visualize import run_dashboard

# # def main():
# #     st.title("Advanced RL Stock Trading")

# #     symbol = st.sidebar.text_input("Symbol:", "TSLA")
# #     start = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
# #     end = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-01-01"))

# #     df = fetch_historical_data(symbol, start, end, interval="1h")
# #     df = add_indicators(df)

# #     if st.sidebar.button("Train New RL Model"):
# #         model = train_rl_agent(df)
# #         st.success("Model trained and saved!")
# #     else:
# #         model = load_rl_agent()

# #     env_obs = df.copy()
# #     env_obs['position'] = 0  # start with no position
# #     actions = []

# #     for step in range(len(df)):
# #         obs = np.append(df.iloc[step].values, env_obs['position'].iloc[step])
# #         action = predict_action(model, obs)
# #         actions.append(action - 1)  # (0:sell->-1, 1:hold->0, 2:buy->1)
# #         if step+1 < len(df):
# #             env_obs['position'].iloc[step+1] = actions[-1]

# #     df['signals'] = actions
# #     run_dashboard(df)

# # if __name__ == "__main__":
# #     main()


# #integrated version

# # # main.py
# # import streamlit as st
# # import pandas as pd
# # from data import fetch_historical_data, fetch_realtime_data
# # from indicators import add_indicators
# # from strategy import rule_based_strategy, ml_based_strategy
# # from backtest import run_backtest
# # from model import train_rl_agent, load_rl_agent, predict_action
# # from visualize import run_dashboard
# # from streamlit_autorefresh import st_autorefresh
# # import numpy as np


# # def main():
# #     # Auto-refresh every 5 seconds during market hours, change as needed
# #     if st.sidebar.checkbox("🔄 Auto-refresh (5-seconds intervals)"):
# #         st_autorefresh(interval=5 * 1000)

# #     st.sidebar.title("⚙️ Trading Pipeline Settings")

# #     symbol = st.sidebar.text_input("Symbol:", "TSLA")

# #     mode = st.sidebar.radio("Select Data Mode:", ["Historical Daily", "Real-Time Intraday"])

# #     if mode == "Historical Daily":
# #         start = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
# #         end = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-01-01"))
# #         df = fetch_historical_data(symbol, start, end, interval="1d")
# #     else:
# #         period = st.sidebar.selectbox("Intraday Period:", ["1d", "5d"])
# #         interval = st.sidebar.selectbox("Intraday Interval:", ["1m", "5m", "15m"])
# #         df = fetch_realtime_data(symbol, period=period, interval=interval)

# #     if df.empty:
# #         st.error("❌ No data returned from Yahoo Finance. Try a longer period or different interval.")
# #         return

# #     df = add_indicators(df)

# #     strategy_choice = st.sidebar.selectbox("Choose Strategy", ["Rule-Based", "ML-Based", "RL Agent"])

# #     if strategy_choice == "Rule-Based":
# #         signals = rule_based_strategy(df)
# #         results = run_backtest(df, signals)
# #         run_dashboard(results, signals, df)

# #     elif strategy_choice == "ML-Based":
# #         signals = ml_based_strategy(df)
# #         results = run_backtest(df, signals)
# #         run_dashboard(results, signals, df)

# #     else:  # RL Agent
# #         if st.sidebar.button("Train New RL Model"):
# #             model = train_rl_agent(df)
# #             st.success("Model trained and saved!")
# #         else:
# #             model = load_rl_agent()

# #         env_obs = df.copy()
# #         env_obs['position'] = 0
# #         actions = []

# #         for step in range(len(df)):
# #             obs = np.append(df.iloc[step].values, env_obs['position'].iloc[step])
# #             action = predict_action(model, obs)
# #             actions.append(action - 1)
# #             if step + 1 < len(df):
# #                 env_obs.loc[env_obs.index[step + 1], 'position'] = actions[-1]

# #         df['signals'] = actions
# #         run_dashboard(df)


# # if __name__ == "__main__":
# #     main()

# import streamlit as st
# import pandas as pd
# import numpy as np
# import os
# from data import fetch_historical_data, fetch_realtime_data
# from indicators import add_indicators
# from strategy import rule_based_strategy, ml_based_strategy
# from backtest import run_backtest
# from model import train_rl_agent, load_rl_agent, predict_action, evaluate_prediction_accuracy
# from visualize import run_dashboard
# from streamlit_autorefresh import st_autorefresh

# def main():
#     # Initialize session state variables if they do not exist
#     if "is_training" not in st.session_state:
#         st.session_state["is_training"] = False
#     if "trained_model" not in st.session_state:
#         st.session_state["trained_model"] = None

#     # Only allow auto-refresh if not in training
#     if not st.session_state["is_training"] and st.sidebar.checkbox("🔄 Auto-refresh (5-seconds intervals)"):
#         st_autorefresh(interval=5 * 1000)

#     st.sidebar.title("⚙️ Trading Pipeline Settings")
#     symbol = st.sidebar.text_input("Symbol:", "TSLA")

#     mode = st.sidebar.radio("Select Data Mode:", ["Historical Daily", "Real-Time Intraday"])

#     if mode == "Historical Daily":
#         start = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
#         end = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-01-01"))
#         df = fetch_historical_data(symbol, start, end, interval="1d")
#     else:
#         period = st.sidebar.selectbox("Intraday Period:", ["1d", "5d"])
#         interval = st.sidebar.selectbox("Intraday Interval:", ["1m", "5m", "15m"])
#         df = fetch_realtime_data(symbol, period=period, interval=interval)


    
#     if df.empty:
#         st.error("❌ No data returned from Yahoo Finance. Try a longer period or a different interval.")
#         return

#     # Add a slider to allow tuning of the fractional order parameter
#     frac_order = st.sidebar.slider("Fractional Order", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

#     # Pass the fractional order into add_indicators
#     df = add_indicators(df, frac_order=frac_order)


 
#     strategy_choice = st.sidebar.selectbox("Choose Strategy", ["Rule-Based", "ML-Based", "RL Agent"])

#     if strategy_choice == "Rule-Based":
#         signals = rule_based_strategy(df)
#         results = run_backtest(df, signals)
#         run_dashboard(results, signals, df)

#     elif strategy_choice == "ML-Based":
#         signals = ml_based_strategy(df)
#         results = run_backtest(df, signals)
#         run_dashboard(results, signals, df)

#     else:  # RL Agent
#         st.sidebar.markdown("---")
#         st.sidebar.markdown("📈 RL Training Options")
#         timesteps = st.sidebar.number_input("Timesteps", value=200000, step=1000)
#         learning_rate = st.sidebar.number_input("Learning Rate", value=0.0003, format="%.6f", step=0.0001)
#         model_path = "trained_models/ppo_trading_agent.zip"
#         training_placeholder = st.empty()

#         # Training is triggered only when the button is pressed.
#         if st.sidebar.button("Train New RL Model"):
#             st.session_state["is_training"] = True  # Mark training as in progress
#             with st.spinner("Training RL Agent..."):
#                 st.session_state["trained_model"] = train_rl_agent(
#                     df, timesteps=timesteps, learning_rate=learning_rate, live_placeholder=training_placeholder
#                 )
#             st.session_state["is_training"] = False  # Training complete
#             st.success("✅ RL Model trained and saved!")
#         else:
#             # If no button press and model file exists, load it.
#             if os.path.exists(model_path):
#                 st.session_state["trained_model"] = load_rl_agent(model_path)
#             else:
#                 st.warning("No trained model found! Please click 'Train New RL Model' to start training.")
#                 st.stop()  # Stop further execution until training occurs

#         model = st.session_state["trained_model"]

#         # Run predictions with the loaded or newly trained model
#         env_obs = df.copy()
#         env_obs['position'] = 0
#         actions = []
#         for step in range(len(df)):
#             obs = np.append(df.iloc[step].values, env_obs['position'].iloc[step])
#             action = predict_action(model, obs)
#             actions.append(action - 1)
#             if step + 1 < len(df):
#                 env_obs.loc[env_obs.index[step + 1], 'position'] = actions[-1]

#         df['signals'] = actions

#         # Evaluate prediction accuracy
#         accuracy, df = evaluate_prediction_accuracy(df)
#         st.metric("📊 Prediction Accuracy (%)", f"{accuracy * 100:.2f}")

#         run_dashboard(df)

# if __name__ == "__main__":
#     try:
#         main()
#     except RuntimeError as e:
#         if "__path__._path" in str(e):
#             print("⚠️ Torch class inspection error suppressed.")
#         else:
#             raise


# main.py

import streamlit as st
import pandas as pd
import numpy as np
import os
from data import fetch_historical_data, fetch_intraday_data
from indicators import add_indicators
from strategy import rule_based_strategy, ml_based_strategy
from backtest import run_backtest
from model import train_rl_agent, load_rl_agent, predict_action, evaluate_prediction_accuracy
from visualize import run_dashboard
from streamlit_autorefresh import st_autorefresh

def main():
    # Initialize session state variables if not already set.
    if "is_training" not in st.session_state:
        st.session_state["is_training"] = False
    if "trained_model" not in st.session_state:
        st.session_state["trained_model"] = None

    # Auto-refresh block (only if not training).
    if not st.session_state["is_training"] and st.sidebar.checkbox("🔄 Auto-refresh (5-seconds intervals)"):
        st_autorefresh(interval=5 * 1000)

    st.sidebar.title("⚙️ Trading Pipeline Settings")
    symbol = st.sidebar.text_input("Symbol:", "TSLA")

    # Change mode names to differentiate between daily and intraday.
    mode = st.sidebar.radio("Select Data Mode:", ["Daily (Historical)", "Intraday (Historical)"])

    if mode == "Daily (Historical)":
        start = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
        end = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-01-01"))
        daily_interval = st.sidebar.selectbox("Daily Interval", ["1d", "1wk", "1mo"])
        df = fetch_historical_data(symbol, start, end, interval=daily_interval)
        # For daily data, use these file names:
        model_path = "trained_models/ppo_trading_agent_daily.zip"
        feature_csv = "trained_models/training_features_daily.csv"
    else:
        period = st.sidebar.selectbox("Intraday Period:", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])
        intraday_interval = st.sidebar.selectbox("Intraday Interval:", ["1m", "5m", "15m", "30m", "60m"])
        df = fetch_intraday_data(symbol, period=period, interval=intraday_interval)
        # For intraday data, use these file names:
        model_path = "trained_models/ppo_trading_agent_intraday.zip"
        feature_csv = "trained_models/training_features_intraday.csv"

    if df.empty:
        st.error("❌ No data returned from Yahoo Finance. Try a different symbol, period, or interval.")
        return

    # Optional slider for tuning the fractional order indicator.
    frac_order = st.sidebar.slider("Fractional Order", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    df = add_indicators(df, frac_order=frac_order)
    
    strategy_choice = st.sidebar.selectbox("Choose Strategy", ["Rule-Based", "ML-Based", "RL Agent"])

    if strategy_choice == "Rule-Based":
        signals = rule_based_strategy(df)
        results = run_backtest(df, signals)
        run_dashboard(results, signals, df)
    elif strategy_choice == "ML-Based":
        signals = ml_based_strategy(df)
        results = run_backtest(df, signals)
        run_dashboard(results, signals, df)
    else:  # RL Agent
        st.sidebar.markdown("---")
        st.sidebar.markdown("📈 RL Training Options")
        timesteps = st.sidebar.number_input("Timesteps", value=200000, step=1000)
        learning_rate = st.sidebar.number_input("Learning Rate", value=0.0003, format="%.6f", step=0.0001)
        training_placeholder = st.empty()

        # If the training button is clicked, train the model and save with the proper file names.
        if st.sidebar.button("Train New RL Model"):
            st.session_state["is_training"] = True
            with st.spinner("Training RL Agent..."):
                st.session_state["trained_model"] = train_rl_agent(
                    df,
                    timesteps=timesteps,
                    learning_rate=learning_rate,
                    live_placeholder=training_placeholder,
                    feature_path=feature_csv,
                    model_save_path=model_path
                )
            st.session_state["is_training"] = False
            st.success("✅ RL Model trained and saved!")
        else:
            # If no button press, load the model if it exists.
            if os.path.exists(model_path):
                st.session_state["trained_model"] = load_rl_agent(model_path=model_path, feature_path=feature_csv)
            else:
                st.warning("No trained model found! Please click 'Train New RL Model' to start training.")
                st.stop()

        model = st.session_state["trained_model"]

        # Generate signals using the loaded or newly trained model.
        env_obs = df.copy()
        env_obs['position'] = 0
        actions = []
        for step in range(len(df)):
            obs = np.append(df.iloc[step].values, env_obs['position'].iloc[step])
            action = predict_action(model, obs)
            actions.append(action - 1)  # Map PPO actions [0,1,2] to positions [-1,0,1]
            if step + 1 < len(df):
                env_obs.loc[env_obs.index[step + 1], 'position'] = actions[-1]
        df['signals'] = actions

        # Evaluate prediction accuracy.
        accuracy, df = evaluate_prediction_accuracy(df)
        st.metric("📊 Prediction Accuracy (%)", f"{accuracy * 100:.2f}")
        run_dashboard(df)

if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        if "__path__._path" in str(e):
            print("⚠️ Torch class inspection error suppressed.")
        else:
            raise
