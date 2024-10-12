import gym
from gym import spaces
import numpy as np

class JengaEnv(gym.Env):
    def __init__(self):
        # Define observation space
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    
        # Define action space
        self.action_space = spaces.Discrete(2)  # 2 discrete actions
    
        # Set initial state
        self.state = 0
    
    def step(self, action):
        # Update state based on action
        if action == 0:
            self.state = max(0, self.state - 1)
        elif action == 1:
            self.state = min(2, self.state + 1)
        
        # Calculate reward
        if self.state == 0:
            reward = -1
        elif self.state == 1:
            reward = 0
        else:
            reward = 1
        
        # Set done flag
        done = False
        
        # Return observation, reward, done, and info
        return self.state, reward, done, {}
    
    def reset(self):
        # Reset environment to initial state
        self.state = 0
        return self.state
    
    def render(self, mode='human'):
        # Render environment state (optional)
        pass

# Register the custom environment
gym.register(id='MyEnvironment-v0', entry_point='my_environment_module:MyEnvironment')
