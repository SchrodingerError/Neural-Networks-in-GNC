import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing

# Define PendulumController class
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
        return self.net(x)  

# ODE solver (RK4 method)
def pendulum_ode_step(state, dt, desired_theta, controller):
    theta, omega, alpha = state

    def compute_torque(th, om, al):
        inp = torch.tensor([[th, om, al, desired_theta]], dtype=torch.float32)
        with torch.no_grad():
            torque = controller(inp)
            torque = torch.clamp(torque, -250, 250)
        return float(torque)

    def derivatives(state, torque):
        th, om, al = state
        a = (g / R) * np.sin(th) + torque / (m * R**2)
        return np.array([om, a, 0])  # dtheta, domega, dalpha

    # Compute RK4 steps
    torque1 = compute_torque(theta, omega, alpha)
    k1 = dt * derivatives(state, torque1)

    state_k2 = state + 0.5 * k1
    torque2 = compute_torque(state_k2[0], state_k2[1], state_k2[2])
    k2 = dt * derivatives(state_k2, torque2)

    state_k3 = state + 0.5 * k2
    torque3 = compute_torque(state_k3[0], state_k3[1], state_k3[2])
    k3 = dt * derivatives(state_k3, torque3)

    state_k4 = state + k3
    torque4 = compute_torque(state_k4[0], state_k4[1], state_k4[2])
    k4 = dt * derivatives(state_k4, torque4)

    new_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    return new_state

# Constants
g = 9.81  # Gravity
R = 1.0   # Length of the pendulum
m = 1.0   # Mass
dt = 0.02  # Time step
num_steps = 500  # Simulation time steps

# Directory containing controller files
controller_dir = "/home/judson/Neural-Networks-in-GNC/inverted_pendulum/training/no_time_weight/controllers"
controller_files = sorted([f for f in os.listdir(controller_dir) if f.startswith("controller_") and f.endswith(".pth")])

# Sorting controllers by epoch
controller_epochs = [int(f.split('_')[1].split('.')[0]) for f in controller_files]
sorted_controllers = [x for _, x in sorted(zip(controller_epochs, controller_files))]

# **Granularity Control: Select every Nth controller**
N = 5  # Change this value to adjust granularity (e.g., every 5th controller)
selected_controllers = sorted_controllers[::N]  # Take every Nth controller

# Initial condition
theta0, omega0, alpha0, desired_theta = (-np.pi, -np.pi, 0.0, np.pi / 6)  # Example initial condition

# Function to run a single controller simulation (for multiprocessing)
def run_simulation(controller_file):
    epoch = int(controller_file.split('_')[1].split('.')[0])
    
    # Load controller
    controller = PendulumController()
    controller.load_state_dict(torch.load(os.path.join(controller_dir, controller_file)))
    controller.eval()
    
    # Run simulation
    state = np.array([theta0, omega0, alpha0])
    theta_vals = []

    for _ in range(num_steps):
        theta_vals.append(state[0])
        state = pendulum_ode_step(state, dt, desired_theta, controller)

    return epoch, theta_vals

# Parallel processing
if __name__ == "__main__":
    num_workers = min(multiprocessing.cpu_count(), 16)  # Limit to 16 workers max
    print(f"Using {num_workers} parallel workers...")
    print(f"Processing every {N}th controller, total controllers used: {len(selected_controllers)}")

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(run_simulation, selected_controllers)

    # Sort results by epoch
    results.sort(key=lambda x: x[0])
    epochs, theta_over_epochs = zip(*results)

    # Convert results to NumPy arrays
    theta_over_epochs = np.array(theta_over_epochs)

    # Create 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Meshgrid for 3D plotting
    E, T = np.meshgrid(epochs, np.arange(num_steps) * dt)

    # Plot surface
    ax.plot_surface(E, T, theta_over_epochs.T, cmap="viridis")

    # Labels
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time (s)")
    ax.set_zlabel("Theta (rad)")
    ax.set_title(f"Pendulum Angle Evolution Over Training Epochs (Granularity N={N})")

    plt.savefig("pendulum_plot.png", dpi=1000, bbox_inches="tight")
    print("Saved plot as 'pendulum_plot.png'.")
