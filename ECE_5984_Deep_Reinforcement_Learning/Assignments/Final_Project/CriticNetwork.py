# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:40:36 2023

@author: agarc
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CriticNetwork(object):
    def __init__(self, state_dim, action_dim, learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        self.model = self.create_critic_network()
        self.target_model = self.create_critic_network()
        self.target_model.set_weights(self.model.get_weights())

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def create_critic_network(self):
        state_input = layers.Input(shape=(self.state_dim,))
        action_input = layers.Input(shape=(self.action_dim,))
        merged = layers.concatenate([state_input, action_input])
        x = layers.Dense(64, activation='relu')(merged)
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(1)(x)
        model = keras.Model(inputs=[state_input, action_input], outputs=output)
        return model

    def train(self, states, actions, q_targets):
        with tf.GradientTape() as tape:
            q_values = self.model([states, actions])
            loss = tf.keras.losses.MSE(q_targets, q_values)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def predict(self, state, action):
        return self.model.predict([state, action])

    def predict_target(self, state, action):
        return self.target_model.predict([state, action])

    def update_target_network(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = 0.005 * critic_weights[i] + (1 - 0.005) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)
