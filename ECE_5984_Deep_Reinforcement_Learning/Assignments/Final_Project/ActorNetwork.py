# -*- coding: utf-8 -*-
"""
Created on Thu May  4 13:36:34 2023

@author: agarc
"""

import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers
from keras import backend as K

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, action_high, action_low, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_high = action_high
        self.action_low = action_low

        K.set_session(sess)

        #Now create the model
        self.model, self.weights, self.state = self.create_actor_network(state_size, action_size)   
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size) 

        #Define action gradient
        self.action_gradient = tf.placeholder(tf.float32, [None, action_size])

        #Define loss and optimize
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = optimizers.Adam(lr=LEARNING_RATE).apply_gradients(grads)

        self.sess.run(tf.global_variables_initializer())

    def create_actor_network(self, state_size, action_size):
        S = layers.Input(shape=[state_size])
        h0 = layers.Dense(units=128, activation='relu')(S)
        h1 = layers.Dense(units=256, activation='relu')(h0)
        h2 = layers.Dense(units=128, activation='relu')(h1)
        RawAction = layers.Dense(units=action_size, activation='tanh')(h2)
        Action = layers.Lambda(lambda x: x * (self.action_high - self.action_low) / 2 + (self.action_high + self.action_low) / 2, name='Action')(RawAction)
        model = models.Model(inputs=S, outputs=Action)
        return model, model.trainable_weights, S

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)
