import torch
import torch.nn as nn

class PendulumDynamics(nn.Module):
    def __init__(self, controller, m:'float'=1, R:'float'=1, g:'float'=9.81):
        super().__init__()
        self.controller = controller
        self.m: 'float' = m
        self.R: 'float' = R
        self.g: 'float' = g

    def forward(self, t, state):
        # Get the current values from the state
        theta, omega, alpha, desired_theta = state[:, 0], state[:, 1], state[:, 2], state[:, 3]

        # Make the input stack for the controller
        input = torch.stack([theta, omega, alpha, desired_theta], dim=1)

        # Get the torque (the output of the neural network)
        tau = self.controller(input).squeeze(-1)

        # Relax alpha
        alpha_desired = (self.g / self.R) * torch.sin(theta) + tau / (self.m * self.R**2)
        dalpha = alpha_desired - alpha
        
        return torch.stack([omega, alpha, dalpha, torch.zeros_like(desired_theta)], dim=1)