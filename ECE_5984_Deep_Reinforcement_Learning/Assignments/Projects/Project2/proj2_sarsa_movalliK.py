# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 04:04:29 2023

@author: kayle
"""

import gym
import collections
import numpy as np
import time
#%%
env = gym.make("FrozenLake-v1", is_slippery = False)
gamma = 0.9
alpha = 0.2
test_epsoids = 200
#%% Create the Object
class Agent_SARSA:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.env = gym.make('FrozenLake-v1', is_slippery = False)
        self.state = self.env.reset()[0]
        self.values = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        
    def sample_env(self):
        """
        Inputs:
        - self: an agent
        Returns:- a tuple: (old_state, action, reward, new_reward)
        """
        old_state = self.state
        action = np.random.choice(range(self.env.action_space.n))
        new_state, reward, done, _, info, = self.env.step(action)
        return old_state, action, reward, new_state, done

    def choose_action(self, state, episode):
        """
        Inputs:
        - self: an agent
        - state: current state
        Returns:
        - next_a: the next action taken.
        """
        if episode < 25:
            return np.random.choice(self.env.action_space.n)
        
        
        if episode > 100:
            self.epsilon = 0.9 * self.epsilon
        
        
        if np.random.uniform(0,1) < self.epsilon:
            action = np.random.choice(self.env.action_space.n)
        else:
            #next_action = [self.values[state, a] for a in range(self.env.action_space.n)]
            #action = np.argmax(next_action)
            action = np.random.choice(np.where(self.values[state,:]==np.max(self.values[state,:]))[0])
        print(action)
        return action

    def value_update(self, s, a, r, next_s, next_a):
        """
        Inputs:
        - self: an agent
        - s: state
        - a: action
        - r: reward
        - next_s: next state
        Returns:
        - self.values[(s, a)]: the updated value of (s, a).
        """
        td_target = r + gamma * self.values[next_s, next_a]
        td_error = td_target - self.values[s, a]
        self.values[s, a] += alpha * td_error
        return self.values[s, a]

    def play_episode(self, env, episode):
        """
        Inputs:
        - self: an agent
        - env: the environment
        Returns:
        - total_reward: the total reward after playing an
        episode
        """
        total_reward = 0
        self.state = self.env.reset()[0]
        action = self.choose_action(self.state, episode)
        done = False
        while not done:
            old_state, action, reward, new_state, done = self.sample_env()
            next_action = self.choose_action(new_state, episode)
            self.value_update(old_state, action, reward, new_state, next_action)
            self.state = new_state
            action = next_action
            total_reward += reward
        return total_reward

agent = Agent_SARSA(epsilon = 0.2)
#%% Run the Episodes
for i in range(test_epsoids):
    episode_reward = agent.play_episode(env = env, episode = i)
    print("Episode:", i+1, "Total Reward:", episode_reward)
#%% Get the Policy

policy = np.zeros(env.observation_space.n)

for i in range(env.observation_space.n):
    policy[i]=np.random.choice(np.where(agent.values[i]==np.max(agent.values[i]))[0])
print(f'Policy: {policy}')
env.close()

#%% Render Policy
env_render = gym.make("FrozenLake-v1", is_slippery = False, render_mode="human")
env_render.reset()

# Run Through with policy
action = int(policy[0])
print(f'Action {action}')
return_value = env_render.step(action)
for i in range(10):
    action = int(policy[return_value[0]])
    return_value = env_render.step(action)
    print(f'Action {action}')
    env_render.render()
    time.sleep(.5)
    if return_value[2]: # break if fallen into hole
        break
env_render.close()