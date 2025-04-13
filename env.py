# env.py
from gymnasium import gym
from gym import spaces
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df
        self.current_step = 0
        self.done = False
        self.position = 0  # -1: short, 0: hold, 1: long
        self.initial_balance = 10000
        self.balance = self.initial_balance

        # Define action space: Sell, Hold, Buy
        self.action_space = spaces.Discrete(3)

        # Define observation space: indicators + current position
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(df.shape[1]+1,), dtype=np.float32)

    def _get_obs(self):
        return np.append(self.df.iloc[self.current_step].values, self.position)

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.done = False
        return self._get_obs()

    def step(self, action):
        reward = 0
        prev_price = self.df['Close'].iloc[self.current_step]

        # Advance the step
        self.current_step += 1
        if self.current_step >= len(self.df)-1:
            self.done = True

        current_price = self.df['Close'].iloc[self.current_step]

        # Execute action
        if action == 0:  # Sell
            reward = self.position * (current_price - prev_price)
            self.position = -1
        elif action == 2:  # Buy
            reward = self.position * (current_price - prev_price)
            self.position = 1
        else:  # Hold
            reward = self.position * (current_price - prev_price)

        self.balance += reward
        info = {'balance': self.balance}
        
        return self._get_obs(), reward, self.done, info
