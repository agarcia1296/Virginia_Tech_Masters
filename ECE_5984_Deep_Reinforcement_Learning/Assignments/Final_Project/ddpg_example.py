# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 02:21:38 2023

@author: froot
"""

import pybullet as p
import pybullet_data
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
#import torch.nn as nn

from RobotArmEnv import RobotArmEnv
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork

# Define the actor-critic network
class Actor(Model):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = layers.Dense(state_size, activation="relu")
        self.fc2 = layers.Dense(state_size, activation="relu")
        self.out = layers.Dense(action_size, activation="tanh")

    def __call__(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x

class Critic(Model):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = layers.Dense(state_size, activation="relu")
        self.fc2 = layers.Dense(state_size, activation="relu")
        self.out = layers.Dense(1, activation=None)

    def __call__(self, state, action):
        x = layers.concatenate([state, action])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x

#Initialize the DDPG agent
class DDPGAgent:
    def __init__(self, state_size, action_size):
        self.actor = Actor(state_size, action_size)
        self.target_actor = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)
        self.target_critic = Critic(state_size, action_size)
        self.actor_optimizer = Adam(learning_rate=0.001)
        self.critic_optimizer = Adam(learning_rate=0.001)
        self.tau = 0.001
        self.gamma = 0.99

    def get_action(self, state):
        state = np.reshape(state, [1,-1])
        return self.actor(state).numpy()[0]
    
    def train(self, states, actions, rewards, next_states, dones):
        # Convert lists to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
    
        # Update the critic
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states)
            q_targets = self.target_critic(next_states, target_actions)
            y = rewards + self.gamma * q_targets * (1 - dones)
            q_values = self.critic(states, actions)
            critic_loss = tf.reduce_mean(tf.square(y - q_values))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
    
        # Update the actor
        with tf.GradientTape() as tape:
            actions_pred = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic(states, actions_pred))
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
    
        # Update the target networks
        self.target_actor.set_weights(self.tau * np.array(self.actor.get_weights()) + (1 - self.tau) * np.array(self.target_actor.get_weights()))
        self.target_critic.set_weights(self.tau * np.array(self.critic.get_weights()) + (1 - self.tau) * np.array(self.target_critic.get_weights()))
        
        
 
#%%

#Train the agent
env = RobotArmEnv()
env.reset()
env.define_goal(2, np.array([0,0.5,0]))
agent = DDPGAgent(env.observation_space.shape[0], env.action_space.shape[0])
episodes = 1000
for episode in range(episodes):
    print(f"Episode: {episode}")
    state = env.reset()
    done = False
    itr = 0
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.train([state], [action], [reward], [next_state], [done])
        state = next_state
        print(f"Episode: {episode} Iteration: {itr} Reward: {reward:.4f}")
        itr = itr + 1

#%%
#Test the agent
state = env.reset()
done = False
while not done:
    action = agent.get_action(state)
    next_state, reward, done = env.step(action)
    state = next_state