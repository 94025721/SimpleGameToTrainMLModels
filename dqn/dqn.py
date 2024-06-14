import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Deep Q-Network (DQN) model definition.
    """

    def __init__(self, state_size, action_size):
        """
        Initialize the DQN model with input and output sizes.

        Args:
            state_size (int): The size of the input state.
            action_size (int): The size of the output action space.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        """
        Define the forward pass of the DQN model.

        Args:
            x (torch.Tensor): Input tensor representing the state.

        Returns:
            torch.Tensor: Output tensor representing Q-values for each action.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

