# -*- coding: utf-8 -*-
"""
Created on Mon May 1 01:42:07 2023

@author: kayle
 
"""

import gym 
import collections
import numpy as np 
import time


ACTIONS = {0: "south"
           ,1: "north"
           ,2: "east"
           ,3: "west"
           ,4: "pickup"
           ,5: "dropoff"}

ENV_NAME = "Taxi-v3"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 2000
EPSILON = 0.9

class Agent_SARSA:
    def __init__(self, gamma, alpha, epsilon):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.env = gym.make(ENV_NAME, render_mode="human")
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
        new_state, reward, terminal_state, info, _ = self.env.step(action)
        return old_state, action, reward, new_state, terminal_state    
        
        
    def choose_action(self, state, episode_num):
        """
        Inputs:
            - self: an agent
            - state: current state
        Returns:
            - next_a: the next action taken
        """
        if episode_num < 10:
            return np.random.choice(self.env.action_space.n)
        
        random_number = np.random.uniform(0,1)
        
        if episode_num > 100:
            self.epsilon = 0.9*self.epsilon
            
        #print(self.epsilon)
        if  random_number < self.epsilon:
            #next_action = [self.values[(state, a)] for a in range(self.env.action_space.n)]
            #next_action = np.argmax(next_action)
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
        
        error = reward + GAMMA * self.values[next_state,next_action] - self.values[state,action]
        self.values[state,action] = self.values[state,action]+ALPHA*error

        
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
        
        
        
    """ POLICY"""
    for i in range(agent_sarsa.env.observation_space.n):
        agent_sarsa.policy[i]=np.random.choice(np.where(agent_sarsa.values[i]==np.max(agent_sarsa.values[i]))[0])
    print(f'Policy: {agent_sarsa.policy}')
    agent_sarsa.env.close()
    """ POLICY"""
    
    
    
    env = gym.make(ENV_NAME, render_mode = 'human')
    env.reset()
    # Run Through with policy
    action = int(agent_sarsa.policy[0])
    return_value = env.step(action)
    while True:
        action = int(agent_sarsa.policy[return_value[0]])
        return_value = env.step(action)
        env.render()
        print(f'Iteration {i} - Action {ACTIONS[action]}')
        time.sleep(.5)
        if return_value[2]: # break if fallen into hole
            break
    env.close()


if __name__ == '__main__':
    main()