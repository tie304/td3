import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, conv_layers):  # TODO re-write for multi and single dim input
        super(Actor, self).__init__()
        self.conv_layers = conv_layers
        if conv_layers:
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
            self.layer_1 = nn.Linear(12 * 21 * 21, 400)
            self.layer_2 = nn.Linear(400, 300)
            self.layer_3 = nn.Linear(300, action_dim)
            self.max_action = max_action
        else:
            self.layer_1 = nn.Linear(state_dim, 400)
            self.layer_2 = nn.Linear(400, 300)
            self.layer_3 = nn.Linear(300, action_dim)
            self.max_action = max_action

    def forward(self, x):

        if self.conv_layers:
            x = self.conv1(x)
            x = F.relu(x)
            x = F.max_pool2d(x,kernel_size=2, stride=2)

            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

            x = x.reshape(-1, 12 * 21 * 21)
            x = F.relu(self.layer_1(x))
            x = F.relu(self.layer_2(x))
            x = self.max_action * torch.tanh(self.layer_3(x))
        else:
            x = F.relu(self.layer_1(x))
            x = F.relu(self.layer_2(x))
            x = self.max_action * torch.tanh(self.layer_3(x))

        return x