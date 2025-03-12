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
base_controller_path = f"/home/judson/Neural-Networks-in-GNC/inverted_pendulum/training/controller_base.pth"

# Initial conditions (theta0, omega0, alpha0, desired_theta)
from initial_conditions import initial_conditions
state_0 = torch.tensor(initial_conditions, dtype=torch.float32, device=device)

# Constants
m = 10.0
g = 9.81
R = 1.0

# Time grid
t_start, t_end, t_points = 0, 10, 1000
t_span = torch.linspace(t_start, t_end, t_points, device=device)

# Specify directory for storing results
output_dir = "loss_function_powers"
os.makedirs(output_dir, exist_ok=True)

# Optimizer values
learning_rate = 1e-1
weight_decay = 1e-4

# Training parameters
num_epochs = 1000

# Define different loss functions based on theta
def make_loss_fn(loss_fn):
    def loss_fn_wrapper(state_traj, t_span):
        theta = state_traj[:, :, 0]            # Extract theta [batch_size, t_points]
        desired_theta = state_traj[:, :, 3]    # Extract desired theta

        # Compute loss using the provided function
        return loss_fn(theta, desired_theta)

    return loss_fn_wrapper


# Define different loss functions based on transformations of theta
loss_functions = {
    'abs_theta': {
        'function': lambda theta, desired_theta: torch.mean(torch.abs(theta - desired_theta)),
        'description': 'Loss is the absolute difference between theta and desired theta'
    },
    'theta_squared': {
        'function': lambda theta, desired_theta: torch.mean(torch.abs((theta - desired_theta) ** 2)),
        'description': 'Loss is the squared difference'
    },
    'theta_cubed': {
        'function': lambda theta, desired_theta: torch.mean(torch.abs((theta - desired_theta) ** 3)),
        'description': 'Loss is the cubed difference'
    },
    'neg_abs_theta': {
        'function': lambda theta, desired_theta: torch.mean(torch.abs(-torch.abs(theta - desired_theta))),
        'description': 'Loss is the absolute negative absolute difference'
    },
    'inverse_theta': {
        'function': lambda theta, desired_theta: torch.mean(torch.abs(1 / (torch.abs(theta - desired_theta) + 1e-6))),
        'description': 'Loss is the inverse absolute difference'
    },
    'inverse_theta_squared': {
        'function': lambda theta, desired_theta: torch.mean(torch.abs(1 / ((theta - desired_theta) ** 2 + 1e-6))),
        'description': 'Loss is the inverse squared difference'
    },
    'inverse_theta_cubed': {
        'function': lambda theta, desired_theta: torch.mean(torch.abs(1 / ((theta - desired_theta) ** 3 + 1e-6))),
        'description': 'Loss is the inverse cubed difference'
    },
}




# Training loop for each loss function
for name, loss_info in loss_functions.items():
    controller = PendulumController().to(device)
    controller.load_state_dict(torch.load(base_controller_path))
    pendulum_dynamics = PendulumDynamics(controller, m, R, g).to(device)
    print(f"Loaded {base_controller_path} as base controller")

    optimizer = optim.Adam(controller.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = make_loss_fn(loss_info['function'])

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
        f.write(f"Base controller path: {base_controller_path}\n")
        f.write(f"Time Span: {t_start} to {t_end}, Points: {t_points}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Weight Decay: {weight_decay}\n")
        f.write("\nLoss Function:\n")
        f.write(inspect.getsource(loss_fn))
        f.write(f"\nLoss Description: {loss_info['description']}\n")
        f.write("\nTraining Cases:\n")
        f.write("[theta0, omega0, alpha0, desired_theta]\n")
        for case in state_0.cpu().numpy():
            f.write(f"{case.tolist()}\n")

    with open(log_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Epoch", "Loss"])
    
    # Training loop
    for epoch in range(0, num_epochs+1):
        optimizer.zero_grad()
        state_traj = odeint(pendulum_dynamics, state_0, t_span, method='rk4')
        loss = loss_fn(state_traj, t_span)
        loss.backward()

        # Save the model before training on this epoch
        # Therefore, controller_epoch represents the controller after {epoch} training iterations
        model_file = os.path.join(controllers_dir, f"controller_{epoch}.pth")
        torch.save(controller.state_dict(), model_file)
        print(f"{model_file} saved with loss: {loss}")

        # Update the weights and biases
        optimizer.step()

        # Logging
        with open(log_file, "a", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([epoch, loss.item()])

print("Training complete. Models and logs are saved under respective directories for each loss function.")