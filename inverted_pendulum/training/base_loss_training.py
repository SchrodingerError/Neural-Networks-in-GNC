import torch
import torch.optim as optim
from torchdiffeq import odeint
import os
import shutil
import csv
import inspect

from PendulumController import PendulumController
from PendulumDynamics import PendulumDynamics
from initial_conditions import initial_conditions
from base_loss_functions import base_loss_functions  # Import the base loss functions

# Device setup
device = torch.device("cpu")
base_controller_path = "/home/judson/Neural-Networks-in-GNC/inverted_pendulum/training/controller_base.pth"

# Initial conditions (theta0, omega0, alpha0, desired_theta)
state_0 = torch.tensor(initial_conditions, dtype=torch.float32, device=device)

# Pendulum constants
m = 10.0
g = 9.81
R = 1.0

# Time grid settings
t_start, t_end, t_points = 0, 10, 1000
t_span = torch.linspace(t_start, t_end, t_points, device=device)

# Directory for storing results
output_dir = "base_loss_training"
os.makedirs(output_dir, exist_ok=True)

# Optimizer hyperparameters
learning_rate = 1e-1
weight_decay = 1e-4

# Training parameters
num_epochs = 1000

# Iterate over the base loss functions.
# Each entry in base_loss_functions is a tuple: (exponent, loss_fn)
for name, (exponent, loss_fn) in base_loss_functions.items():
    
    # Create a wrapper loss function that applies the base loss function
    # to the extracted theta and desired_theta from the state trajectory,
    # and then reduces it to a scalar.
    def current_loss_fn(state_traj):
        theta = state_traj[:, :, 0]         # [batch_size, t_points]
        desired_theta = state_traj[:, :, 3]   # [batch_size, t_points]
        return torch.mean(loss_fn(theta, desired_theta))
    
    # Initialize the controller and load the base parameters.
    controller = PendulumController().to(device)
    controller.load_state_dict(torch.load(base_controller_path))
    pendulum_dynamics = PendulumDynamics(controller, m, R, g).to(device)
    print(f"Loaded base controller from {base_controller_path} for loss '{name}' (exponent {exponent})")

    optimizer = optim.Adam(controller.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Set up directories for saving models and logs for this loss function.
    function_output_dir = os.path.join(output_dir, name)
    controllers_dir = os.path.join(function_output_dir, "controllers")
    if os.path.exists(controllers_dir):
        shutil.rmtree(controllers_dir)
    os.makedirs(controllers_dir, exist_ok=True)

    config_file = os.path.join(function_output_dir, "training_config.txt")
    log_file = os.path.join(function_output_dir, "training_log.csv")

    # Save configuration details including the loss function's exponent and source code.
    with open(config_file, "w") as f:
        f.write(f"Base controller path: {base_controller_path}\n")
        f.write(f"Time Span: {t_start} to {t_end}, Points: {t_points}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Weight Decay: {weight_decay}\n")
        f.write(f"\nLoss Function Name: {name}\n")
        f.write(f"Loss Function Exponent: {exponent}\n")
        f.write("\nLoss Function Source Code:\n")
        f.write(inspect.getsource(loss_fn))
        f.write("\nTraining Cases:\n")
        f.write("[theta0, omega0, alpha0, desired_theta]\n")
        for case in state_0.cpu().numpy():
            f.write(f"{case.tolist()}\n")

    # Create log file with header.
    with open(log_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Epoch", "Loss"])

    # Begin training loop.
    for epoch in range(num_epochs + 1):
        optimizer.zero_grad()
        state_traj = odeint(pendulum_dynamics, state_0, t_span, method='rk4')
        loss = current_loss_fn(state_traj)
        loss.backward()

        # Save the model at this epoch.
        model_file = os.path.join(controllers_dir, f"controller_{epoch}.pth")
        torch.save(controller.state_dict(), model_file)
        print(f"{model_file} saved with loss: {loss.item()}")

        optimizer.step()

        # Log the training progress.
        with open(log_file, "a", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([epoch, loss.item()])

print("Training complete. Models and logs are saved under respective directories for each loss function.")
