# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 20:14:49 2023

@author: agarc
"""

import gym 
import numpy as np 
import time


ACTIONS = {0: "left"
           ,1: "down"
           ,2: "right"
           ,3: "up"}

def main(): 
    policy_list = []
    for gamma in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0]:
        print(f"gamma = {gamma}")
        env = gym.make('FrozenLake-v1', render_mode = 'human') # or the latest version
        env.reset()
        optimal_value_function = value_iteration(env, gamma);
        optimal_policy = extract_policy(env, optimal_value_function, gamma)
        policy_list.append(optimal_policy)
        print(f'Policy: {optimal_policy}')
        
        # Run Through with policy
        max_iterations = 1000
        action = int(optimal_policy[0])
        return_value = env.step(action)
        for i in range(max_iterations):
            action = int(optimal_policy[return_value[0]])
            return_value = env.step(action)
            env.render()
            print(f'Iteration {i} - Action {ACTIONS[action]}')
            time.sleep(.5)
            if return_value[2]: # break if fallen into hole
                break
            
        env.close()


def extract_policy(env, v, gamma = 1.0):
    """
     Inputs:
     - value_table: state value function
     - gamma: discount factor
    Returns:
     - policy: the optimal policy
    """
    policy = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy


def value_iteration(env, gamma = 1.0):
    """ 
    Inputs: 
        - env: the frozen lake environment. 
        - gamma: discount factor 
 
    Returns: 
        - value_table: state value function 
        - Q_value: state-action value function (Q function)  
    """ 
    value_table = np.zeros(env.observation_space.n)  # initialize value-function
    theta = 1e-20
    while True:
        prev_value = np.copy(value_table)
        for s in range(env.observation_space.n):
            q_sa = [sum([p*(r + prev_value[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.action_space.n)] 
            value_table[s] = max(q_sa)
        if (np.sum(np.fabs(prev_value - value_table)) <= theta):
            break
    return value_table

if __name__ == '__main__':
    main()




         
     