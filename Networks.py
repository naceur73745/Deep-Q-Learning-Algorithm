# Importing necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim

# Class to represent a Q-network for reinforcement learning
class Qnetwork(nn.Module):
    def __init__(self, input_dim, fc1_dim, fc2_dim, n_action, lr):
        """
        Initialize the Qnetwork with specified parameters.

        Parameters:
        - input_dim: Dimension of the input state
        - fc1_dim: Dimension of the first fully connected layer
        - fc2_dim: Dimension of the second fully connected layer
        - n_action: Number of possible actions
        - lr: Learning rate for the optimizer
        """
        super(Qnetwork, self).__init__()

        # Define the neural network architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, fc1_dim),
            nn.Tanh(),
            nn.Linear(fc1_dim, fc2_dim),
            nn.Tanh(),
            nn.Linear(fc2_dim, n_action),
            nn.Softmax(dim=-1)
        )

        # Set the learning rate and initialize the optimizer
        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Define the loss function
        self.loss = nn.MSELoss()

    def forward(self, state):
        """
        Forward pass through the Q-network.

        Parameters:
        - state: Input state for which Q-values are to be predicted

        Returns:
        - actions: Predicted Q-values for each action in the input state
        """
        actions = self.network(state)
        return actions
