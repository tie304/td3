import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, conv_layers):  # TODO re-write for multi and single dim input
        super(Critic, self).__init__()
        self.conv_layers = conv_layers
        if conv_layers:
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
            # Defining the first Critic neural network
            self.layer_1 = nn.Linear((12 * 21 * 21) + action_dim, 400)
            self.layer_2 = nn.Linear(400, 300)
            self.layer_3 = nn.Linear(300, 1)
            # Defining the second Critic neural network
            self.layer_4 = nn.Linear((12 * 21 * 21) + action_dim, 400)
            self.layer_5 = nn.Linear(400, 300)
            self.layer_6 = nn.Linear(300, 1)
        else:
            # Defining the first Critic neural network
            self.layer_1 = nn.Linear(state_dim + action_dim, 400)
            self.layer_2 = nn.Linear(400, 300)
            self.layer_3 = nn.Linear(300, 1)
            # Defining the second Critic neural network
            self.layer_4 = nn.Linear(state_dim + action_dim, 400)
            self.layer_5 = nn.Linear(400, 300)
            self.layer_6 = nn.Linear(300, 1)

    def forward(self, x, u):  # TODO re-write for multi and single dim input
        if self.conv_layers:
            # first conv block
            x = self.conv1(x)
            x = F.relu(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = x.reshape(-1, 12 * 21 * 21) # TODO needs to follow conv size algorithm

        xu = torch.cat([x, u], 1)
        # Forward-Propagation on the first Critic Neural Network
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        # Forward-Propagation on the second Critic Neural Network
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2

    def Q1(self, x, u):  # TODO re-write for multi and single dim input
        if self.conv_layers:
            x = self.conv1(x)
            x = F.relu(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = x.reshape(-1, 12 * 21 * 21)

        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1
