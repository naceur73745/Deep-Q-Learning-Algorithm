# Importing necessary libraries and modules
from ReplayBuffer import ReplayBuffer
from Networks import Qnetwork
import numpy as np
import torch
import random

# Class to represent a reinforcement learning agent
class Agent:
    def __init__(self, input_dim, fc1_dim, fc2_dim, n_actions, lr, batch_size, mem_size, gamma, epsilon_dec):
        """
        Initialize the Agent with specified parameters.

        Parameters:
        - input_dim: Dimension of the input state
        - fc1_dim: Dimension of the first fully connected layer in the neural network
        - fc2_dim: Dimension of the second fully connected layer in the neural network
        - n_actions: Number of possible actions
        - lr: Learning rate for the neural network optimizer
        - batch_size: Size of the batch for learning
        - mem_size: Size of the replay buffer memory
        - gamma: Discount factor for future rewards
        - epsilon_dec: Rate of epsilon decay for exploration-exploitation trade-off
        """
        self.input_dim = input_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.n_action = n_actions
        self.lr = lr
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.epsilon = 1  # Initial epsilon for exploration-exploitation trade-off
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = 0.01  # Minimum value for epsilon
        self.gamma = gamma

        # Initialize replay buffer memory and neural network
        self.mem = ReplayBuffer(mem_size, input_dim, n_actions, batch_size)
        self.network = Qnetwork(input_dim, fc1_dim, fc2_dim, n_actions, lr)

    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy strategy.

        Parameters:
        - state: Current state for which action is to be chosen

        Returns:
        - action: Chosen action
        """
        # Choose an action between exploration and exploitation
        threshold = random.uniform(0, 1)

        if threshold > self.epsilon:
            # Exploitation: Choose the action with the highest predicted Q-value
            state = torch.tensor(state)
            actions = self.network.forward(state)
            action = torch.argmax(actions).item()
        else:
            # Exploration: Sample an action randomly from the action space
            action = random.randint(0, self.n_action - 1)

        return action

    def learn(self):
        """
        Update the Q-network based on sampled experiences from the replay buffer.
        """
        # Check if the memory is sufficiently filled
        if self.batch_size < self.mem.mem_cntr:
            return

        # Sample a batch of experiences from the replay buffer
        states, new_state, action, reward, done, batch_indices = self.mem.sample_mem()

        # Convert to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32)
        new_state = torch.tensor(new_state)
        reward = torch.tensor(reward)
        action = torch.tensor(action, dtype=torch.int32)
        done = torch.tensor(done)

        # Assign each state-action pair to a corresponding Q-value and find the predicted Q-values
        index = np.arange(self.batch_size, dtype=np.int32)
        q_predicts = self.network.forward(states)[index, action]

        # Find the Q-values for the next states
        q_next = self.network.forward(new_state)
        q_target = reward + self.gamma * torch.max(q_next, dim=1)[0]  # Max Q-value for each next state

        # Compute the loss and perform a backward pass
        loss = self.network.loss(q_target, q_predicts)
        loss.backward()

        # Update the Q-network parameters using the optimizer
        self.network.optimizer.step()

        # Decay epsilon for exploration-exploitation trade-off
        self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_min)
