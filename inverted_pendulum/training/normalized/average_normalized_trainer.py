import torch
import torch.optim as optim
from torchdiffeq import odeint
import numpy as np
import os
import shutil
import csv
import inspect

from PendulumController import PendulumController
from PendulumDynamics import PendulumDynamics

# Device setup
device = torch.device("cpu")

# Initial conditions (theta0, omega0, alpha0, desired_theta)
from initial_conditions import initial_conditions
state_0 = torch.tensor(initial_conditions, dtype=torch.float32, device=device)

# Device setup
device = torch.device("cpu")

# Constants
m = 10.0
g = 9.81
R = 1.0

# Time grid
t_start, t_end, t_points = 0, 10, 1000
t_span = torch.linspace(t_start, t_end, t_points, device=device)

# Specify directory for storing results
output_dir = "average_normalized"
os.makedirs(output_dir, exist_ok=True)

# Use a previously generated random seed
random_seed = 4529

# Set the seeds for reproducibility
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Print the chosen random seed
print(f"Random seed for torch and numpy: {random_seed}")

# Initialize controller and dynamics
controller = PendulumController().to(device)
pendulum_dynamics = PendulumDynamics(controller, m, R, g).to(device)

# Optimizer setup
learning_rate = 1e-1
weight_decay = 1e-4
optimizer = optim.Adam(controller.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Training parameters
num_epochs = 1001

# Define loss functions
def make_loss_fn(weight_fn):
    def loss_fn(state_traj, t_span):
        theta = state_traj[:, :, 0]            # Size: [batch_size, t_points]
        desired_theta = state_traj[:, :, 3]    # Size: [batch_size, t_points]
        weights = weight_fn(t_span)            # Initially Size: [t_points]

        # Reshape or expand weights to match theta dimensions
        weights = weights.view(-1, 1)  # Now Size: [batch_size, t_points]

        # Calculate the weighted loss
        return torch.mean(weights * (theta - desired_theta) ** 2)

    return loss_fn

# Define and store weight functions with descriptions, normalized by average weight
weight_functions = {
    'constant': {
        'function': lambda t: torch.ones_like(t) / torch.ones_like(t).mean(),
        'description': 'Constant weight: All weights are 1, normalized by the average (remains 1)'
    },
    'linear': {
        'function': lambda t: (t / t.max()) / (t / t.max()).mean(),
        'description': 'Linear weight: Weights increase linearly from 0 to 1, normalized by the average weight'
    },
    'quadratic': {
        'function': lambda t: ((t / t.max()) ** 2) / ((t / t.max()) ** 2).mean(),
        'description': 'Quadratic weight: Weights increase quadratically from 0 to 1, normalized by the average weight'
    },
    'exponential': {
        'function': lambda t: (torch.exp(t / t.max() * 2)) / (torch.exp(t / t.max() * 2)).mean(),
        'description': 'Exponential weight: Weights increase exponentially, normalized by the average weight'
    },
    'inverse': {
        'function': lambda t: (1 / (t / t.max() + 1)) / (1 / (t / t.max() + 1)).mean(),
        'description': 'Inverse weight: Weights decrease inversely, normalized by the average weight'
    },
    'inverse_squared': {
        'function': lambda t: (1 / ((t / t.max() + 1) ** 2)) / (1 / ((t / t.max() + 1) ** 2)).mean(),
        'description': 'Inverse squared weight: Weights decrease inversely squared, normalized by the average weight'
    }
}

# Training loop for each weight function
for name, weight_info in weight_functions.items():
    controller = PendulumController().to(device)
    pendulum_dynamics = PendulumDynamics(controller, m, R, g).to(device)
    optimizer = optim.Adam(controller.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = make_loss_fn(weight_info['function'])

    # File paths
    function_output_dir = os.path.join(output_dir, name)
    controllers_dir = os.path.join(function_output_dir, "controllers")

    # Check if controllers directory exists and remove it
    if os.path.exists(controllers_dir):
        shutil.rmtree(controllers_dir)
    os.makedirs(controllers_dir, exist_ok=True)

    config_file = os.path.join(function_output_dir, "training_config.txt")
    log_file = os.path.join(function_output_dir, "training_log.csv")

    # Overwrite configuration and log files
    with open(config_file, "w") as f:
        f.write(f"Random Seed: {random_seed}\n")
        f.write(f"Time Span: {t_start} to {t_end}, Points: {t_points}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Weight Decay: {weight_decay}\n")
        f.write("\nLoss Function:\n")
        f.write(inspect.getsource(loss_fn))
        f.write("\nTraining Cases:\n")
        f.write("[theta0, omega0, alpha0, desired_theta]\n")
        for case in state_0.cpu().numpy():
            f.write(f"{case.tolist()}\n")

    with open(log_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Epoch", "Loss"])

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        state_traj = odeint(pendulum_dynamics, state_0, t_span, method='rk4')
        loss = loss_fn(state_traj, t_span)
        loss.backward()
        optimizer.step()

        # Logging
        with open(log_file, "a", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([epoch, loss.item()])

        # Save the model
        model_file = os.path.join(controllers_dir, f"controller_{epoch}.pth")
        torch.save(controller.state_dict(), model_file)
        print(f"{model_file} saved with loss: {loss}")

print("Training complete. Models and logs are saved under respective directories for each loss function.")