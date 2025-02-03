import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
# 1) 3D Controller: [theta, omega, alpha] -> torque
# ----------------------------------------------------------------
class PendulumController3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_3d):
        """
        x_4d: shape (batch_size, 4) => [theta, cos(theta), omega, alpha].
        Returns shape: (batch_size, 1) => torque.
        """
        raw_torque = self.net(x_3d)
        clamped_torque = torch.clamp(raw_torque, -250, 250)  # Clamp torque within [-250, 250]
        return clamped_torque


# ----------------------------------------------------------------
# 2) Define ODE System Using `odeint`
# ----------------------------------------------------------------
m = 10.0
g = 9.81
R = 1.0

class PendulumDynamics3D(nn.Module):
    """
    Defines the ODE system for [theta, omega, alpha] with torque tracking.
    """

    def __init__(self, controller):
        super().__init__()
        self.controller = controller

    def forward(self, t, state):
        """
        state: (batch_size, 4) => [theta, omega, alpha, tau_prev]
        Returns: (batch_size, 4) => [dtheta/dt, domega/dt, dalpha/dt, dtau/dt]
        """

        theta = state[:, 0]
        omega = state[:, 1]
        alpha = state[:, 2]
        tau_prev = state[:, 3]
        
        # Create tensor input for controller: [theta, omega, alpha]
        input_3d = torch.stack([theta, omega, alpha], dim=1)  # shape (batch_size, 3)

        # Compute torque using the controller
        tau = self.controller(input_3d).squeeze(-1)  # shape (batch_size,)

        # Compute desired alpha
        alpha_desired = (g / R) * torch.sin(theta) + tau / (m * R**2)

        # Define ODE system
        dtheta = omega
        domega = alpha
        dalpha = alpha_desired - alpha  # Relaxation term
        dtau = tau - tau_prev  # Keep track of torque evolution

        return torch.stack([dtheta, domega, dalpha, dtau], dim=1)  # (batch_size, 4)

# ----------------------------------------------------------------
# 3) Loss Function
# ----------------------------------------------------------------
def loss_fn(state_traj, t_span):
    """
    Computes loss based on the trajectory with inverse time weighting (1/t) for theta and omega.

    Args:
        state_traj: Tensor of shape (time_steps, batch_size, 4).
        t_span: Tensor of time steps (time_steps,).

    Returns:
        total_loss, (loss_theta, loss_omega)
    """
    theta = state_traj[:, :, 0]  # (time_steps, batch_size)

    loss_theta = 1e3 * torch.mean(theta**2)

    # Combine the losses
    total_loss = loss_theta

    return total_loss, (loss_theta)

# ----------------------------------------------------------------
# 4) Training Setup
# ----------------------------------------------------------------
device = torch.device("cpu")

# Create the controller and pendulum dynamics model
controller = PendulumController3D().to(device)
pendulum_dynamics = PendulumDynamics3D(controller).to(device)

# Define optimizer
optimizer = optim.Adam(controller.parameters(), lr=1e-1)

# Initial conditions: [theta, omega, alpha, tau_prev]
initial_conditions = [
    [0.1,  0.0, 0.0, 0.0],   # Small perturbation
    [-0.5,  0.0, 0.0, 0.0],  
    [6.28,  6.28, 0.0, 0.0],  
    [1.57, 0.5, 0.0, 0.0],  
    [0.0, -6.28, 0.0, 0.0],  
    [1.57, -6.28, 0.0, 0.0],
]

# Convert to torch tensor (batch_size, 4)
state_0 = torch.tensor(initial_conditions, dtype=torch.float32, device=device)

# Time grid
t_span = torch.linspace(0, 10, 200, device=device)

num_epochs = 100_000
print_every = 25

# ----------------------------------------------------------------
# 5) Training Loop
# ----------------------------------------------------------------
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Integrate the ODE
    state_traj = odeint(pendulum_dynamics, state_0, t_span, method='rk4')  
    # state_traj shape: (time_steps, batch_size, 4)

    # Compute loss
    total_loss, (l_theta) = loss_fn(state_traj, t_span)

    # Check for NaN values
    if torch.isnan(total_loss):
        print(f"NaN detected at epoch {epoch}. Skipping step.")
        optimizer.zero_grad()
        continue  # Skip this iteration

    # Backprop
    total_loss.backward()
    optimizer.step()


    # Print progress
    if epoch % print_every == 0:
        print(f"Epoch {epoch:4d}/{num_epochs} | "
              f"Total: {total_loss.item():.6f} | "
              f"Theta: {l_theta.item():.6f}")

        torch.save(controller.state_dict(), "controller_cpu_clamped_no_time_penalty.pth")
        print("Model saved as 'controller_cpu_clamped_no_time_penalty.pth'.")
