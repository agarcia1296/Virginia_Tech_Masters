import numpy as np

class agent_qlearning:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, num_actions=4):
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.num_actions = num_actions  # number of possible actions
        self.q_table = {}  # Q-value table to store Q-values for each state-action pair
    
    def act(self, state):
        # Choose an action using an epsilon-greedy policy
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            q_values = self.q_table.get(state, np.zeros(self.num_actions))
            action = np.argmax(q_values)
        return action
    
    def learn(self, state, action, reward, next_state, done):
        # Update Q-value for the state-action pair using Q-learning update rule
        q_values = self.q_table.get(state, np.zeros(self.num_actions))
        next_q_values = self.q_table.get(next_state, np.zeros(self.num_actions))
        max_next_q = np.max(next_q_values)
        td_target = reward + self.gamma * max_next_q * (1 - done)
        td_error = td_target - q_values[action]
        q_values[action] += self.alpha * td_error
        self.q_table[state] = q_values
