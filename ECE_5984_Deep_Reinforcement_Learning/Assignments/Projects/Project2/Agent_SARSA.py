# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 00:29:06 2023

@author: agarc
"""

import gym 
import collections
import numpy as np 
import time


ACTIONS = {0: "left"
           ,1: "down"
           ,2: "right"
           ,3: "up"}

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 200
EPSILON = 0.2

class Agent_SARSA:
    def __init__(self, gamma, alpha, epsilon):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.env = gym.make(ENV_NAME, desc = None, map_name = '4x4', is_slippery = False)
        self.state = self.env.reset()[0]
        self.values = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        #self.values = collections.defaultdict(float)
        self.policy = np.zeros(self.env.observation_space.n)

    def sample_env(self):
        '''
        Inputs:
            - self: an agent
        Returns:
            - a tuple: (old_state, action, reward, new_reward)
        '''
        old_state = self.state
        action = np.random.choice(range(self.env.action_space.n))
        new_state, reward, terminal_state, _, _ = self.env.step(action)
        return old_state, action, reward, new_state, terminal_state    
        
        
    def choose_action(self, state, episode_num):
        """
        Inputs:
            - self: an agent
            - state: current state
        Returns:
            - next_a: the next action taken
        """
        if episode_num < 20:
            return np.random.choice(self.env.action_space.n)
        
        random_number = np.random.uniform(0,1)
        
        if episode_num > 100:
            self.epsilon = 0.9*self.epsilon
            
        #print(self.epsilon)
        if  random_number < self.epsilon:
            next_action = np.random.choice(self.env.action_space.n)
        else:
            next_action = np.random.choice(np.where(self.values[state,:]==np.max(self.values[state,:]))[0])
        return next_action
        

    def value_update(self, state, action, reward, next_state, next_action):
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
        #td_target = reward + GAMMA * self.values[(next_state, next_action)]
        #td_error = td_target - self.values[(state, action)]
        #self.values[(state, action)] += ALPHA * td_error
        
        error = reward + self.gamma* self.values[next_state,next_action] - self.values[state,action]
        self.values[state,action] = self.values[state,action]+self.alpha*error

        
        return self.values[state, action]        
        

    def play_episode(self, episode_num):
        """
        Inputs:
            - self: an agent
            - env: the environment
        Returns:
            - total_reward: the total reward after playing an episode
        """
        total_reward = 0
        self.state = self.env.reset()[0]
        action = self.choose_action(self.state, episode_num)
        terminal_state = False
        while not terminal_state:
            old_state, action, reward, new_state, terminal_state = self.sample_env()
            next_action = self.choose_action(new_state, episode_num)
            self.value_update(old_state, action, reward, new_state, next_action)
            self.state = new_state
            action = next_action
            total_reward += reward
            if terminal_state:
                break
   
        return total_reward
    
    

def main():
    agent_sarsa = Agent_SARSA(GAMMA, ALPHA, EPSILON)
    for episode in range(TEST_EPISODES):
        reward = agent_sarsa.play_episode(episode)
        print("Episode:", episode+1, "Total Reward:", reward)
        
    for i in range(agent_sarsa.env.observation_space.n):
        agent_sarsa.policy[i]=np.random.choice(np.where(agent_sarsa.values[i]==np.max(agent_sarsa.values[i]))[0])
    print(f'Policy: {agent_sarsa.policy}')
    agent_sarsa.env.close()
    
    env = gym.make(ENV_NAME, desc = None, map_name = '4x4', is_slippery = False, render_mode = 'human')
    env.reset()
    # Run Through with policy
    action = int(agent_sarsa.policy[0])
    print(f'Action {ACTIONS[action]}')
    return_value = env.step(action)
    for i in range(10):
        action = int(agent_sarsa.policy[return_value[0]])
        return_value = env.step(action)
        env.render()
        print(f'Action {ACTIONS[action]}')
        time.sleep(.5)
        if return_value[2]: # break if fallen into hole
            break
    env.close()


if __name__ == '__main__':
    main()