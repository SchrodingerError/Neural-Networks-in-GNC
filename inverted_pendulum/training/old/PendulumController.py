import torch
import torch.nn as nn

class PendulumController(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        raw_torque = self.net(x)
        return torch.clamp(raw_torque, -250, 250)