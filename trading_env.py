import gymnasium as gym
import numpy as np
from gymnasium import spaces

class StockTradingEnv(gym.Env):
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.current_step = 0
        
        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # State: Last 5 days prices + 1 Holding status (Total 6 values)
        self.window_size = 5
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.window_size + 1,), dtype=np.float32)
        
        self.holdings = 0 # 0 = No stock, 1 = Have stock

    def reset(self, seed=None, options=None):
        self.current_step = self.window_size
        self.holdings = 0
        return self._next_observation(), {}

    def _next_observation(self):
        # Pichle 5 din ka data
        frame = self.df['Normalized'].iloc[self.current_step - self.window_size : self.current_step].values
        # Current holding status add karo
        obs = np.append(frame, [self.holdings])
        return obs.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        reward = 0
        done = self.current_step >= len(self.df) - 1
        
        current_price = self.df['Normalized'].iloc[self.current_step]
        prev_price = self.df['Normalized'].iloc[self.current_step - 1]
        
        # Reward Logic
        if action == 1: # Buy
            if self.holdings == 0:
                self.holdings = 1
        elif action == 2: # Sell
            if self.holdings == 1:
                self.holdings = 0
                reward = (current_price - prev_price) * 100 # Profit/Loss magnitude badhane ke liye
        elif action == 0: # Hold
             if self.holdings == 1:
                 reward = (current_price - prev_price) * 10 # Paper profit
        
        return self._next_observation(), reward, done, False, {}