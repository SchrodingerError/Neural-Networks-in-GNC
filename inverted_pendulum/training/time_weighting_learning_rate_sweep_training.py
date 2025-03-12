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

from time_weighting_functions import weight_functions

# Device and base path setup
device = torch.device("cpu")
base_controller_path = f"/home/judson/Neural-Networks-in-GNC/inverted_pendulum/training/controller_base.pth"

# Initial conditions
from initial_conditions import initial_conditions
state_0 = torch.tensor(initial_conditions, dtype=torch.float32, device=device)

# Constants
m = 10.0
g = 9.81
R = 1.0

# Time grid
t_start, t_end, t_points = 0, 10, 1000
t_span = torch.linspace(t_start, t_end, t_points, device=device)

# Output directory setup
base_output_dir = "time_weighting_learning_rate_sweep"
os.makedirs(base_output_dir, exist_ok=True)

# Weight decay
weight_decay = 0

# Training parameters
num_epochs = 200
# Learning rates for the sweep
learning_rates = [16, 8, 4, 2, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.16, 0.125, 0.1, 0.08, 0.05, 0.04, 0.02, 0.01]

# Define loss function
def make_loss_fn(weight_fn):
    def loss_fn(state_traj, t_span):
        theta = state_traj[:, :, 0]            # Size: [batch_size, t_points]
        desired_theta = state_traj[:, :, 3]    # Size: [batch_size, t_points]
        
        min_weight = 0.01  # Weights are on the range [min_weight, 1]
        weights = weight_fn(t_span, min_val=min_weight)            # Initially Size: [t_points]
        # Reshape or expand weights to match theta dimensions
        weights = weights.view(-1, 1)  # Now Size: [batch_size, t_points]

        # Calculate the weighted loss
        return torch.mean(weights * (theta - desired_theta) ** 2)

    return loss_fn

# Training loop for each weight function and learning rate
for name, weight_fn in weight_functions.items():
    for lr in learning_rates:
        output_dir = os.path.join(base_output_dir, f"{name}/lr_{lr:.3f}")
        controllers_dir = os.path.join(output_dir, "controllers")
        if os.path.exists(controllers_dir):
            shutil.rmtree(controllers_dir)
        os.makedirs(controllers_dir, exist_ok=True)

        # Load controller, setup dynamics and optimizer
        controller = PendulumController().to(device)
        controller.load_state_dict(torch.load(base_controller_path))

        pendulum_dynamics = PendulumDynamics(controller, m, R, g).to(device)
        optimizer = optim.Adam(controller.parameters(), lr=lr, weight_decay=weight_decay)

        loss_fn = make_loss_fn(weight_fn)

        # Configuration and log files
        config_file = os.path.join(output_dir, "training_config.txt")
        log_file = os.path.join(output_dir, "training_log.csv")
        with open(config_file, "w") as f:
            f.write(f"Base controller path: {base_controller_path}\n")
            f.write(f"Time Span: {t_start} to {t_end}, Points: {t_points}\n")
            f.write(f"Learning Rate: {lr}\n")
            f.write(f"Weight Decay: {weight_decay}\n")
            f.write("\nLoss Function:\n")
            f.write(inspect.getsource(loss_fn))
            f.write("\nWeight Function:\n")
            f.write(inspect.getsource(weight_fn))
            f.write("\nTraining Cases:\n")
            f.write("[theta0, omega0, alpha0, desired_theta]\n")
            for case in state_0.cpu().numpy():
                f.write(f"{case.tolist()}\n")

        with open(log_file, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Epoch", "Loss"])

        # Track loss and NaN detection
        previous_losses = []
        nan_counter = 0
        stop_count = 5  # Number of epochs to check for unchanged loss or consecutive NaN

        # Training loop
        for epoch in range(num_epochs+1):
            optimizer.zero_grad()
            state_traj = odeint(pendulum_dynamics, state_0, t_span, method='rk4')
            loss = loss_fn(state_traj, t_span)
            
            if torch.isnan(loss).item():
                nan_counter += 1
                if nan_counter >= stop_count:
                    print(f"Consecutive NaN detected for {stop_count} epochs at epoch {epoch}. Terminating training for {name} with learning rate {lr}.")
                    break
            else:
                nan_counter = 0  # Reset NaN counter if no NaN detected

            loss.backward()
            optimizer.step()

            # Save the model
            model_file = os.path.join(controllers_dir, f"controller_{epoch}.pth")
            torch.save(controller.state_dict(), model_file)
            print(f"{model_file} saved with loss: {loss}")

            with open(log_file, "a", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([epoch, loss.item()])

            # Early stopping if loss does not change
            if len(previous_losses) >= stop_count:
                if all(abs(prev_loss - loss.item()) < 1e-6 for prev_loss in previous_losses[-stop_count:]):
                    print(f"Loss unchanged for {stop_count} epochs at epoch {epoch}. Terminating training for {name} with learning rate {lr}.")
                    break
            previous_losses.append(loss.item())

print("Training complete. Models and logs are saved under respective directories for each learning rate and weight function.")