import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import numpy as np
import random

# Generate a random seed
random_seed = random.randint(0, 10000)

# Set the seeds for reproducibility
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Print the chosen random seed
print(f"Random seed for torch and numpy: {random_seed}")

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
        return torch.clamp(raw_torque, -250, 250)  # Clamp torque

# Constants
m = 10.0
g = 9.81
R = 1.0

class PendulumDynamics(nn.Module):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

    def forward(self, t, state):
        theta = state[:, 0]
        omega = state[:, 1]
        alpha = state[:, 2]
        desired_theta = state[:, 3]  # Extract desired theta

        # Pass desired_theta as input to the controller
        input = torch.stack([theta, omega, alpha, desired_theta], dim=1)
        tau = self.controller(input).squeeze(-1)

        alpha_desired = (g / R) * torch.sin(theta) + tau / (m * R**2)

        dtheta = omega
        domega = alpha
        dalpha = alpha_desired - alpha

        return torch.stack([dtheta, domega, dalpha, torch.zeros_like(desired_theta)], dim=1)

def loss_fn(state_traj):
    theta = state_traj[:, :, 0]  # Extract theta
    desired_theta = state_traj[:, :, 3]  # Extract desired theta

    loss_theta = 1e3 * torch.mean((theta - desired_theta)**2)
    return loss_theta

# Device setup
device = torch.device("cpu")

# Initial conditions (theta0, omega0, alpha0, desired_theta)
state_0 = torch.tensor([
    # Theta perturbations
    [1/6 * torch.pi,    0.0, 0.0, 0.0],
    [-1/6 * torch.pi,   0.0, 0.0, 0.0],
    [2/3 * torch.pi,    0.0, 0.0, 0.0],
    [-2/3 * torch.pi,   0.0, 0.0, 0.0],

    # Omega perturbations
    [0.0, 1/3 * torch.pi,     0.0, 0.0],
    [0.0, -1/3 * torch.pi,    0.0, 0.0],
    [0.0, 2 * torch.pi,     0.0, 0.0],
    [0.0, -2 * torch.pi,    0.0, 0.0],

    # Return to non-zero theta
    [0.0, 0.0, 0.0,     2*torch.pi],
    [0.0, 0.0, 0.0,     -2*torch.pi],
    [0.0, 0.0, 0.0,     1/2 * torch.pi],
    [0.0, 0.0, 0.0,     -1/2 *torch.pi],
    [0.0, 0.0, 0.0,     1/3 * torch.pi],
    [0.0, 0.0, 0.0,     -1/3 *torch.pi],

    # Mix cases
    [1/4 * torch.pi,    1 * torch.pi,   0.0,    0.0],
    [-1/4 * torch.pi,   -1 * torch.pi,  0.0,    0.0],
    [1/2 * torch.pi,    -1 * torch.pi,  0.0,    1/3 * torch.pi],
    [-1/2 * torch.pi,   1 * torch.pi,   0.0,    -1/3 *torch.pi],
    [1/4 * torch.pi,    1 * torch.pi,   0.0,    2 * torch.pi],
    [-1/4 * torch.pi,   -1 * torch.pi,  0.0,    2 * torch.pi],
    [1/2 * torch.pi,    -1 * torch.pi,  0.0,    4 * torch.pi],
    [-1/2 * torch.pi,   1 * torch.pi,   0.0,    -4 *torch.pi],

], dtype=torch.float32, device=device)

# Time grid
t_span = torch.linspace(0, 10, 1000, device=device)

# Initialize controller and dynamics
controller = PendulumController().to(device)
pendulum_dynamics = PendulumDynamics(controller).to(device)

# Optimizer
optimizer = optim.Adam(controller.parameters(), lr=1e-1, weight_decay=0)

# Training parameters
num_epochs = 10_000
print_every = 25

for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    state_traj = odeint(pendulum_dynamics, state_0, t_span, method='rk4')
    loss = loss_fn(state_traj)

    if torch.isnan(loss):
        print(f"NaN detected at epoch {epoch}. Skipping step.")
        optimizer.zero_grad()
        continue

    loss.backward()
    optimizer.step()

    if epoch % print_every == 0:
        print(f"Epoch {epoch}/{num_epochs} | Loss: {loss.item():.6f}")
        torch.save(controller.state_dict(), "controller_with_desired_theta.pth")
        print("Model saved as 'controller_with_desired_theta.pth'.")
