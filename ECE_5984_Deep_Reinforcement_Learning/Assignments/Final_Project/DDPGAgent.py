
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import RandomUniform
from keras.engine.training import collect_trainable_weights
from keras import layers, models, optimizers
from tensorflow import keras
import random


# Hyperparameters
BUFFER_SIZE = 1000000
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.001
CRITIC_LR = 0.0001
ACTOR_LR = 0.0001
ACTOR_HIDDEN_LAYERS = [256, 128]
CRITIC_HIDDEN_LAYERS = [256, 128]

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



class DDPGAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_max = env.action_space.high[0]
        self.actor_lr = 0.001
        self.critic_lr = 0.002
        self.gamma = 0.99
        self.tau = 0.001
        self.memory = []
        self.batch_size = 128
        
        # Actor
        self.actor_model = self.build_actor_model()
        self.target_actor_model = self.build_actor_model()
        
        # Critic
        self.critic_model = self.build_critic_model()
        self.target_critic_model = self.build_critic_model()
        
        self.target_actor_model.set_weights(self.actor_model.get_weights())
        self.target_critic_model.set_weights(self.critic_model.get_weights())
    
    def build_actor_model(self):
        print('Building Actor Model')
        S = Input(shape=[self.state_dim])
        h1 = Dense(ACTOR_HIDDEN_LAYERS[0], activation='relu')(S)
        h2 = Dense(ACTOR_HIDDEN_LAYERS[1], activation='relu')(h1)
        output = Dense(self.action_dim, activation='tanh')(h2)
        output = output * self.action_max
        model = Model(inputs=S, outputs=output)
        adam = Adam(lr=self.actor_lr)
        model.compile(loss='mse', optimizer=adam)
        print('Done Building Actor Model!')
        return model
    
    def build_critic_model(self):
        print('Building Critic Model')
        S = Input(shape=[self.state_dim])
        state_h1 = Dense(CRITIC_HIDDEN_LAYERS[0], activation='relu')(S)
        state_h2 = Dense(CRITIC_HIDDEN_LAYERS[1])(state_h1)
        action_input = Input(shape=(self.action_dim,))
        action_h1 = Dense(300)(action_input)
        merged = concatenate()([state_h2, action_h1])
        merged_h1 = Dense(300, activation='relu')(merged)
        output = Dense(1, activation='linear')(merged_h1)
        model = Model(inputs=[S, action_input], outputs=output)
        adam = Adam(lr=self.critic_lr)
        model.compile(loss='mse', optimizer=adam)
        print('Done Building Critic Model!')
        return model
    
    def target_train(self):
        actor_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau)* actor_target_weights[i]
        self.target_actor_model.set_weights(actor_target_weights)
    
    
    def act(self, state):
        state = np.reshape(state, [1, self.state_dim])
        action = self.actor_model.predict(state)[0]
        return action
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])
        if len(self.memory) > 100000:
            del self.memory[0]
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        new_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])
        target_actions = self.target_actor_model.predict_on_batch(new_states)
        target_q_values = self.target_critic_model.predict_on_batch([new_states, target_actions])
        target_q_values[dones] = 0 # Set the target Q-value for done states to zero
        # Compute the target Q-values using the Bellman equation
        targets = rewards + self.gamma * target_q_values.flatten()
        # Train the critic model using the sampled batch
        self.critic_model.train_on_batch([states, actions], targets)