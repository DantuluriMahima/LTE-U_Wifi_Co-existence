import random
import numpy as np
import tensorflow as tf
import math
from collections import deque

from Params.ConstantParams import PARAMS

class Learning:
    def __init__(self):
        # Parameters
        self.exploration = 30000  # Exploration iterations
        self.do_dyna = 0
        self.k = 0.05
        self.epsilon = 0.999 # Initial exploration rate
        self.epsilon_min = 0.01  # Minimum epsilon
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.learning_rate = 0.001
        self.gamma = 0.6  # Discount factor
        self.batch_size = 32
        self.memory = deque(maxlen=2000)  # Replay memory
        self.pFactor = [1,1.05,1.10]
        self.originalPower = PARAMS.pTxLTE
        self.currentPfactor = 1
        # Fairness and LTE thresholds
        self.fairness_threshold = 0.8
        self.u_lte_threshold = 0.8
        self.n = 2
        # Initialize Q-network and target network
        self.state_space = self.create_state_space()
        self.initial_state = self.state_space[0]
        self.current_state = self.initial_state
        self.previous_state = None
        self.action_space = self.create_action_space()
        self.current_action = None
        self.action_length = 0
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_network()  # Sync target with Q-network

    def build_model(self):
        """Build the neural network model for the Q-function."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(6,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(self.action_space), activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0),
                      loss='mse')
        return model
    
    def update_target_network(self):
        """Updates the target network with weights from the Q-network."""
        self.target_model.set_weights(self.model.get_weights())

    # Initialize state space and action space
    def create_state_space(self):
        """Initialize the state space."""
        self.state_space = [[l, m, n, p, q, r] for l in range(1,8) for m in range(1,8)
                            for n in range(1,8) for p in range(3) for q in range(3) for r in range(3)]
        return self.state_space
    
    def create_action_space(self):
        """Initialize the action space."""
        self.action_space = []
        actions = [-1, 0, 1]
        for action in actions:
            for var in ['l', 'm', 'n', 'p', 'q', 'r']:
                a = [0, 0, 0, 0, 0, 0]
                idx = ['l', 'm', 'n', 'p', 'q', 'r'].index(var)
                a[idx] = action
                self.action_space.append(tuple(a))
        self.action_space = list(set(self.action_space))
        return self.action_space

    def get_valid_actions(self, state):
        """Get valid actions based on state boundaries."""
        valid_actions = set()
        l, m, n, p, q, r = state
        for action in self.action_space:
            l_new, m_new, n_new, p_new, q_new, r_new = (l + action[0], m + action[1], n + action[2],
                                                        p + action[3], q + action[4], r + action[5])
            if 0 < l_new <= 7 and 0 < m_new <= 7 and 0 < n_new <= 7 and 0 <= p_new <= 2 \
                    and 0 <= q_new <= 2 and 0 <= r_new <= 2:
                valid_actions.add(action)
        self.action_length = len(valid_actions)
        return list(valid_actions)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Epsilon-greedy policy to select an action and update the current state and power factor."""
        self.previous_state = self.current_state
        
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            # Exploration: choose a random valid action
            valid_actions = self.get_valid_actions(state)
            if not valid_actions:
                return None  # Return if no valid actions are available

            # Randomly select one of the valid actions
            action = random.choice(valid_actions)
        else:
            # Exploitation: predict Q-values for the current state and choose the best action
            q_values = self.model.predict(np.array([state]))

            # Get valid actions and mask invalid actions by setting Q-values to -inf
            valid_actions = self.get_valid_actions(state)
            masked_q_values = np.full_like(q_values, -np.inf)
            for a in valid_actions:
                try:
                    idx = self.action_space.index(a)
                    masked_q_values[0][idx] = q_values[0][idx]
                except ValueError:
                    continue

            # Choose the action with the highest Q-value from valid actions
            action_index = np.argmax(masked_q_values[0])
            action = self.action_space[action_index]

        # Update current action and state based on the chosen action
        self.current_action = action
        self.current_state = tuple(a + b for a, b in zip(self.previous_state, self.current_action))

        # Update power factor if any of the indices 3, 4, or 5 changed in the state
        for index in [3, 4, 5]:
            if self.previous_state[index] != self.current_state[index]:
                self.currentPfactor = self.pFactor[self.current_state[index]]
                break

        return action




    def replay(self):
        """Train the Q-network using experience replay."""
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            # Get the target Q-value for the next state
            target = reward if done else reward + self.gamma * np.amax(self.target_model.predict(np.array([next_state]))[0])
            
            # Get Q-values for the current state
            target_f = self.model.predict(np.array([state]))
            
            # Mask invalid actions for the current state
            valid_actions = self.get_valid_actions(state)
            masked_target_f = np.full_like(target_f, -np.inf)  # Mask invalid actions as -inf
            for a in valid_actions:
                try:
                    idx = self.action_space.index(a)
                    masked_target_f[0][idx] = target_f[0][idx]  # Keep Q-values for valid actions
                except ValueError:
                    continue  # Skip invalid actions

            # Update the target Q-value for the chosen action
            action_index = self.action_space.index(action)
            masked_target_f[0][action_index] = target
            
            # Train the model with the masked target Q-values
            self.model.fit(np.array([state]), masked_target_f, epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def RewardFunction(self, Fairness, scene_params):
        if Fairness < self.fairness_threshold:
            reward_fx = (self.fairness_threshold - Fairness)/((1/self.n)-self.fairness_threshold)
        else:
            reward_fx = (Fairness - self.fairness_threshold)/0.2
        
        return reward_fx

    def load(self, name):
        """Load weights from a file."""
        self.model.load_weights(name)

    def save(self, name):
        """Save weights to a file."""
        self.model.save_weights(name)