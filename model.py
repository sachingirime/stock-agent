# # model.py
# import pandas as pd
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# from env import TradingEnv

# def train_rl_agent(df, timesteps=10000):
#     env = DummyVecEnv([lambda: TradingEnv(df)])
#     model = PPO('MlpPolicy', env, verbose=1, device='cpu')

#     model.learn(total_timesteps=timesteps)
#     model.save("trained_models/ppo_trading_agent")
#     return model

# def load_rl_agent(path="trained_models/ppo_trading_agent.zip"):
#     env = DummyVecEnv([lambda: TradingEnv(pd.DataFrame())])  # dummy environment for loading
#     return PPO.load(path, env=env)

# def predict_action(model, obs):
#     action, _states = model.predict(obs)
#     return action

# model.py
import os
import json
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from env import TradingEnv
import plotly.graph_objects as go
import streamlit as st

class MetricsCallback(BaseCallback):
    def __init__(self, save_path="trained_models/training_log.json", log_freq=1, live_placeholder=None):
        super().__init__()
        self.metrics = {
            "loss": [],
            "value_loss": [],
            "policy_gradient_loss": [],
            "approx_kl": [],
            "entropy_loss": [],
            "clip_fraction": [],
            "explained_variance": []
        }
        self.save_path = save_path
        self.log_freq = log_freq
        self.counter = 0
        self.live_placeholder = live_placeholder

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.counter += 1
        info = self.model.logger.name_to_value
        for key in self.metrics.keys():
            value = info.get(f"train/{key}", None)
            if value is not None:
                self.metrics[key].append(float(value))
        if self.counter % self.log_freq == 0:
            log_data = {"metrics": self.metrics}
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, "w") as f:
                json.dump(log_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            if self.live_placeholder:
                tabs = self.live_placeholder.tabs(["Loss", "Policy Gradient", "Value Loss"])
                with tabs[0]:
                    fig = go.Figure(data=go.Scatter(y=self.metrics["loss"], mode='lines+markers'))
                    fig.update_layout(title="Loss Over Time")
                    st.plotly_chart(fig, use_container_width=True)
                with tabs[1]:
                    fig = go.Figure(data=go.Scatter(y=self.metrics["policy_gradient_loss"], mode='lines+markers'))
                    fig.update_layout(title="Policy Gradient Loss")
                    st.plotly_chart(fig, use_container_width=True)
                with tabs[2]:
                    fig = go.Figure(data=go.Scatter(y=self.metrics["value_loss"], mode='lines+markers'))
                    fig.update_layout(title="Value Loss")
                    st.plotly_chart(fig, use_container_width=True)

def train_rl_agent(df, timesteps=50000, learning_rate=0.0003, live_placeholder=None,
                   feature_path="trained_models/training_features.csv", model_save_path="trained_models/ppo_trading_agent.zip"):
    os.makedirs("trained_models", exist_ok=True)
    # Save the training features to the specified CSV file.
    df.to_csv(feature_path, index=False)
    env = DummyVecEnv([lambda: TradingEnv(df)])
    model = PPO('MlpPolicy', env, verbose=1, device='cpu', learning_rate=learning_rate)
    callback = MetricsCallback(save_path="trained_models/training_log.json", log_freq=1, live_placeholder=live_placeholder)
    model.learn(total_timesteps=timesteps, callback=callback)
    model.save(model_save_path)
    return model

def load_rl_agent(model_path="trained_models/ppo_trading_agent.zip", feature_path="trained_models/training_features.csv"):
    # Load the features from the saved CSV file to create a dummy environment.
    if not os.path.exists(feature_path):
        raise FileNotFoundError("Training feature file missing. Train the model first.")
    dummy_df = pd.read_csv(feature_path).iloc[:10].fillna(0)
    from stable_baselines3.common.vec_env import DummyVecEnv
    env = DummyVecEnv([lambda: TradingEnv(dummy_df)])
    return PPO.load(model_path, env=env)

def predict_action(model, obs):
    action, _states = model.predict(obs)
    return action

def evaluate_prediction_accuracy(df):
    df = df.copy()
    df['actual_return'] = df['Close'].shift(-1) - df['Close']
    df['actual_direction'] = df['actual_return'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df['correct'] = (df['signals'] == df['actual_direction']).astype(int)
    filtered = df[df['signals'] != 0]
    if not filtered.empty:
        accuracy = filtered['correct'].mean()
    else:
        accuracy = 0.0
    return accuracy, df
