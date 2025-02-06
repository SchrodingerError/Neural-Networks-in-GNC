import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import numpy as np
import inspect
import time
import csv
import os

# Specify directory for storing results
output_dir = "training/quadratic_time_weight"
controller_output_dir = output_dir + "/controllers"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
os.makedirs(controller_output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Use a previously generated random seed
random_seed = 4529

# Set the seeds for reproducibility
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Print the chosen random seed
print(f"Random seed for torch and numpy: {random_seed}")

# Constants
m = 10.0
g = 9.81
R = 1.0

# Device setup
device = torch.device("cpu")

# Time grid
t_start, t_end, t_points = 0, 10, 1000
t_span = torch.linspace(t_start, t_end, t_points, device=device)

# Initial conditions (theta0, omega0, alpha0, desired_theta)
state_0 = torch.tensor([
    [1/6 * torch.pi, 0.0, 0.0, 0.0],
    [-1/6 * torch.pi, 0.0, 0.0, 0.0],
    [2/3 * torch.pi, 0.0, 0.0, 0.0],
    [-2/3 * torch.pi, 0.0, 0.0, 0.0],
    [0.0, 1/3 * torch.pi, 0.0, 0.0],
    [0.0, -1/3 * torch.pi, 0.0, 0.0],
    [0.0, 2 * torch.pi, 0.0, 0.0],
    [0.0, -2 * torch.pi, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2 * torch.pi],
    [0.0, 0.0, 0.0, -2 * torch.pi],
    [0.0, 0.0, 0.0, 1/2 * torch.pi],
    [0.0, 0.0, 0.0, -1/2 * torch.pi],
    [0.0, 0.0, 0.0, 1/3 * torch.pi],
    [0.0, 0.0, 0.0, -1/3 * torch.pi],
    [1/4 * torch.pi, 1 * torch.pi, 0.0, 0.0],
    [-1/4 * torch.pi, -1 * torch.pi, 0.0, 0.0],
    [1/2 * torch.pi, -1 * torch.pi, 0.0, 1/3 * torch.pi],
    [-1/2 * torch.pi, 1 * torch.pi, 0.0, -1/3 * torch.pi],
    [1/4 * torch.pi, 1 * torch.pi, 0.0, 2 * torch.pi],
    [-1/4 * torch.pi, -1 * torch.pi, 0.0, 2 * torch.pi],
    [1/2 * torch.pi, -1 * torch.pi, 0.0, 4 * torch.pi],
    [-1/2 * torch.pi, 1 * torch.pi, 0.0, -4 * torch.pi],
], dtype=torch.float32, device=device)

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

class PendulumDynamics(nn.Module):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

    def forward(self, t, state):
        theta, omega, alpha, desired_theta = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        input = torch.stack([theta, omega, alpha, desired_theta], dim=1)
        tau = self.controller(input).squeeze(-1)
        alpha_desired = (g / R) * torch.sin(theta) + tau / (m * R**2)
        return torch.stack([omega, alpha, alpha_desired - alpha, torch.zeros_like(desired_theta)], dim=1)

def loss_fn(state_traj, t_span):
    theta = state_traj[:, :, 0]
    desired_theta = state_traj[:, :, 3]

    # Make the time weights broadcastable for theta [len(t_span), len(batches)]
    time_weights = (t_span ** 2).view(-1, 1)

    return 1e3 * torch.mean(time_weights * (theta - desired_theta) ** 2)

# Initialize controller and dynamics
controller = PendulumController().to(device)
pendulum_dynamics = PendulumDynamics(controller).to(device)

# Optimizer setup
learning_rate = 1e-1
weight_decay = 1e-4
optimizer = optim.Adam(controller.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training parameters
num_epochs = 5_001  # 5000 + 1 for 5000 total weight updates
save_every = 1      # How often to save the controller.pth

# File paths
config_file = os.path.join(output_dir, "training_config.txt")
log_file = os.path.join(output_dir, "training_log.csv")
model_file = ""     # Placeholder for the model file, updated in training loop

# Save configuration details
with open(config_file, "w") as f:
    f.write(f"Random Seed: {random_seed}\n")
    f.write(f"Time Span: {t_start} to {t_end}, Points: {t_points}\n")
    f.write(f"Learning Rate: {learning_rate}\n")
    f.write(f"Weight Decay: {weight_decay}\n")
    f.write("\nLoss Function:\n")
    f.write(inspect.getsource(loss_fn))  # Extract and write loss function source code
    f.write("\nTraining Cases:\n")
    f.write("[theta0, omega0, alpha0, desired_theta]\n")
    for case in state_0.cpu().numpy():
        f.write(f"{case.tolist()}\n")


# Overwrite the log file at the start
with open(log_file, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Epoch", "Loss", "Elapsed Time (s)"])

# Training loop with real-time logging and NaN tracking
start_time = time.time()
with open(log_file, "a", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        optimizer.zero_grad()
        state_traj = odeint(pendulum_dynamics, state_0, t_span, method='rk4')
        loss = loss_fn(state_traj, t_span)

        elapsed_time = time.time() - epoch_start_time

        if torch.isnan(loss):
            print(f"NaN detected at epoch {epoch}. Skipping step.")
            csv_writer.writerow([epoch, "NaN detected", elapsed_time])
            csvfile.flush()  # Ensure real-time writing
            optimizer.zero_grad()
            continue

        loss.backward()
        optimizer.step()

        # Log normal loss
        csv_writer.writerow([epoch, loss.item(), elapsed_time])
        csvfile.flush()  # Ensure real-time writing

        if epoch % save_every == 0:
            print(f"Epoch {epoch}/{num_epochs} | Loss: {loss.item():.6f} | Time: {elapsed_time:.4f} sec")
            model_file = os.path.join(controller_output_dir, f"controller_{epoch}.pth")
            torch.save(controller.state_dict(), model_file)

# Final save
torch.save(controller.state_dict(), model_file)
print(f"No time weight training complete. Model and files saved in directory '{output_dir}'. Model saved as '{model_file}'. Logs saved in '{log_file}' and configuration in '{config_file}'.")