# Most of this code was provided in the DRL course, I modified the network structure to experiment with the solution

import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size=8, action_size=4, seed=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Let's define the smallest NN that we can think of
        self.input = nn.Linear(state_size, 128)
        # On My laptop
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, action_size)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.input(state)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.output(x)
        # F.softmax(x, dim=1)
