# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 01:42:07 2023

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


class Agent_QLearning:
    def __init__(self, env_name, alpha, gamma, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = gym.make(env_name, desc = None, map_name = '4x4', is_slippery = False)
        self.state = self.env.reset()[0]
        #self.values = collections.defaultdict(float)
        self.values = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.policy = np.zeros(self.env.observation_space.n)
    
    def sample_env(self):
        '''
        Inputs:
            - self: an agent
        Returns:
            - a tuple: (old_state, action, reward, new_reward)
                    
        if np.max(self.values[self.state]) > 0:
            action = np.argmax(self.values[self.state])
        else:
            action = self.env.action_space.sample()
        '''
        old_state = self.state
        action = self.env.action_space.sample()
        new_state, reward, terminal_state, _, _ = self.env.step(action)
        return old_state, action, reward, new_state, terminal_state

    def best_value_and_action(self, state, episode_num):
        """
        Inputs:
            - self: an agent
            - state: current state
        Returns:
            - best_value: the best value updated
            - best_action: the best action taken
        """
        if episode_num < 25:
            best_action = np.random.choice(range(self.env.action_space.n))
            action_values = self.values[state]
            best_value = action_values[best_action]
            return best_value, best_action
        
        random_number = np.random.uniform(0,1)
        
        if episode_num > 50:
            self.epsilon = 0.9*self.epsilon
            
        #print(self.epsilon)
        if  random_number < self.epsilon:
            best_action = np.random.choice(self.env.action_space.n)
            action_values = self.values[state,:]
            best_value = action_values[best_action]
        else:
            best_action = np.argmax(self.values[state,:])
            action_values = self.values[state,:]
            best_value = action_values[best_action]
        return best_value, best_action
    
    def value_update(self, s, a, r, next_s):
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
        best_next_q = np.max(self.values[next_s,:])
        self.values[s, a] = self.values[s, a] + self.alpha * (r + self.gamma * best_next_q - self.values[s, a])


    def play_episode(self, episode_num):
        """
        Inputs:
            - self: an agent
            - env: the environment
        Returns:
            - total_reward: the total reward after playing an episode
        """
        total_reward = 0.0
        self.state = self.env.reset()[0]
        best_value, best_action = self.best_value_and_action(self.state, episode_num)
        terminal_state = False
        while not terminal_state:
            old_state, action, reward, new_state, terminal_state = self.sample_env()
            best_value, best_action = self.best_value_and_action(old_state, episode_num)
            print(best_value)
            #next_state, reward, terminal_state, _, _ = self.env.step(action)
            total_reward += reward
            self.value_update(old_state, best_action, reward, new_state)
            if terminal_state:
                break
            self.state = new_state
        print(f"Episode: {episode_num} Total reward: {total_reward}")
        return total_reward


def main():
    agent = Agent_QLearning(ENV_NAME, ALPHA, GAMMA, EPSILON)
    rewards = []
    for i in range(TEST_EPISODES):
        rewards.append(agent.play_episode(i))
    print(f"Average reward over {TEST_EPISODES} episodes: {sum(rewards) / TEST_EPISODES}") 
    agent.env.close()

    for i in range(agent.env.observation_space.n):
        agent.policy[i]=np.random.choice(np.where(agent.values[i]==np.max(agent.values[i]))[0])
    print(f'Policy: {agent.policy}')
    agent.env.close()
    
    env = gym.make(ENV_NAME, desc = None, map_name = '4x4', is_slippery = False, render_mode = 'human')
    env.reset()
    # Run Through with policy
    action = int(agent.policy[0])
    print(f'Iteration 0 - Action {ACTIONS[action]}')
    return_value = env.step(action)
    max_itr = 10
    for j in range(max_itr):
        action = int(agent.policy[return_value[0]])
        return_value = env.step(action)
        env.render()
        print(f'Iteration {j+1} - Action {ACTIONS[action]}')
        time.sleep(.5)
        if return_value[2]: # break if fallen into hole
            break
    env.close()


if __name__ == '__main__':
    main()