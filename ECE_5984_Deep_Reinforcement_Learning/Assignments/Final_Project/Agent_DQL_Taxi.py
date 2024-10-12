# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:50:24 2023

@author: kayle
"""

# Deep Q Learning / Frozen Lake / Not Slippery / 4x4
import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings("ignore")
import time
import os
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
from collections import deque

ACTIONS = {0: "south"
           ,1: "north"
           ,2: "east"
           ,3: "west"
           ,4: "pickup"
           ,5: "dropoff"}

ENV_NAME = "Taxi-v3"
TRAIN_EPISODES=4000
TEST_EPISODES=100
max_steps=300
batch_size=32
LEARNRATE = 0.001
MINEPSILON = 0.01
MAXEPSILON = 1
EPSDECAY = 0.001/3
GAMMA = 0.9
EPSILON = 0.9

class Agent_DQL:
    def __init__(self, env_name, gamma, epsilon, mineps, maxeps, epsdecay, learnrate):
        self.memory = deque(maxlen=2500)
        self.gamma = gamma
        self.epsilon = epsilon
        self.env = gym.make(ENV_NAME, render_mode="human")
        self.values = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.epsilon_lst=[]
        self.model = self.buildmodel()

    def buildmodel(self):
        model=Sequential()
        model.add(Dense(10, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def add_memory(self, new_state, reward, done, state, action):
        self.memory.append((new_state, reward, done, state, action))

    def action(self, state):
        if np.random.rand() > self.epsilon:
            return np.random.randint(0,4)
        return np.argmax(self.model.predict(state))

    def pred(self, state):
        return np.argmax(self.model.predict(state))

    def replay(self,batch_size):
        minibatch=random.sample(self.memory, batch_size)
        for new_state, reward, done, state, action in minibatch:
            target= reward
            if not done:
                target=reward + self.gamma* np.amax(self.model.predict(new_state))
            target_f= self.model.predict(state)
            target_f[0][action]= target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.min_eps:
            self.epsilon=(self.max_eps - self.min_eps) * np.exp(-self.eps_decay*episode) + self.min_eps

        self.epsilon_lst.append(self.epsilon)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def main():
    
    agent = Agent_DQL(ENV_NAME, GAMMA, EPSILON, MINEPSILON, MAXEPSILON, EPSDECAY, LEARNRATE)
    
    env = gym.make(ENV_NAME, render_mode = 'human')
    env.reset()
    
    reward_lst=[]
    for episode in range(TRAIN_EPISODES):
        state= env.reset()
        state_arr=np.zeros(env.observation_space.n)
        state_arr[state] = 1
        state= np.reshape(state_arr, [1, env.observation_space.n])
        reward = 0
        done = False
        for t in range(max_steps):
            # env.render()
            action = agent.action(state)
            new_state, reward, done, _, _ = env.step(action)
            new_state_arr = np.zeros(env.observation_space.n)
            new_state_arr[new_state] = 1
            new_state = np.reshape(new_state_arr, [1, env.observation_space.n])
            agent.add_memory(new_state, reward, done, state, action)
            state = new_state
    
            if done:
                print(f'Episode: {episode:4}/{TRAIN_EPISODES} and step: {t:4}. Eps: {float(agent.epsilon):.2}, reward {reward}')
                break
    
        reward_lst.append(reward)
    
        if len(agent.memory)> batch_size:
            agent.replay(batch_size)
    
    print(' Train mean % score= ', round(100*np.mean(reward_lst),1))
    
    # test
    test_wins=[]
    for episode in range(TEST_EPISODES):
        state = env.reset()
        state_arr=np.zeros(env.observation_space.n)
        state_arr[state] = 1
        state= np.reshape(state_arr, [1, env.observation_space.n])
        done = False
        reward=0
        state_lst = []
        state_lst.append(state)
        print('******* EPISODE ',episode, ' *******')
    
        for step in range(max_steps):
            action = agent.pred(state)
            new_state, reward, done, _, _ = env.step(action)
            new_state_arr = np.zeros(env.observation_space.n)
            new_state_arr[new_state] = 1
            new_state = np.reshape(new_state_arr, [1, env.observation_space.n])
            state = new_state
            state_lst.append(state)
            if done:
                print(reward)
                # env.render()
                break
    
        test_wins.append(reward)
    env.close()

if __name__ == '__main__':
    main()