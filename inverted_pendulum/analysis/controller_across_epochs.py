import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool, cpu_count

# Define PendulumController class
from PendulumController import PendulumController

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
m = 10.0   # Mass
dt = 0.02  # Time step
num_steps = 500  # Simulation time steps

# Directory containing controller files
loss_function = "quadratic"
controller_dir = f"/home/judson/Neural-Networks-in-GNC/inverted_pendulum/training/normalized/training/{loss_function}/controllers"
#controller_dir = f"C:/Users/Judson/Desktop/New Gitea/Neural-Networks-in-GNC/inverted_pendulum/training/{loss_function}/controllers"
controller_files = sorted([f for f in os.listdir(controller_dir) if f.startswith("controller_") and f.endswith(".pth")])

# Sorting controllers by epoch
controller_epochs = [int(f.split('_')[1].split('.')[0]) for f in controller_files]
sorted_controllers = [x for _, x in sorted(zip(controller_epochs, controller_files))]

# **Epoch Range Selection**
epoch_range = (0, 1000)  # Set your desired range (e.g., (0, 5000) or (0, 100))

filtered_controllers = [
    f for f in sorted_controllers
    if epoch_range[0] <= int(f.split('_')[1].split('.')[0]) <= epoch_range[1]
]

# **Granularity Control: Select every Nth controller**
N = 1  # Change this value to adjust granularity (e.g., every 5th controller)
selected_controllers = filtered_controllers[::N]  # Take every Nth controller within the range

# Initial condition
# theta0, omega0, alpha0, desired_theta = (-np.pi, -2*np.pi, 0.0, -1.3*np.pi)  # Example initial condition
theta0, omega0, alpha0, desired_theta = (-np.pi, 0.0, 0.0, 0.0)  # Example initial condition

# Parallel function must return epoch explicitly
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

    return epoch, theta_vals  # Return epoch with data

# Parallel processing
if __name__ == "__main__":
    num_workers = min(cpu_count(), 16)  # Limit to 16 workers max
    print(f"Using {num_workers} parallel workers...")
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(run_simulation, selected_controllers)

    # Sort results by epoch to ensure correct order
    results.sort(key=lambda x: x[0])  
    epochs, theta_over_epochs = zip(*results)  # Unzip sorted results

    # Convert results to NumPy arrays
    theta_over_epochs = np.array(theta_over_epochs)

    # Create 3D line plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    time_steps = np.arange(num_steps) * dt  # X-axis (time)

    # Plot each controller as a separate line
    for epoch, theta_vals in zip(epochs, theta_over_epochs):
        ax.plot(
            [epoch] * len(time_steps),  # Y-axis (epoch stays constant for each line)
            time_steps,  # X-axis (time)
            theta_vals,  # Z-axis (theta evolution)
            label=f"Epoch {epoch}" if epoch % (N * 10) == 0 else "",  # Label some lines for clarity
        )

    # Labels
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time (s)")
    ax.set_zlabel("Theta (rad)")
    ax.set_title(f"Pendulum Angle Evolution for {loss_function}")

    # Add a horizontal line at desired_theta across all epochs and time steps
    epochs_array = np.array([epoch for epoch, _ in zip(epochs, theta_over_epochs)])
    ax.plot(
        epochs_array,  # X-axis spanning all epochs
        [time_steps.max()] * len(epochs_array),  # Y-axis at the maximum time step
        [desired_theta] * len(epochs_array),  # Constant Z-axis value of desired_theta
        color='r', linestyle='--', linewidth=2, label='Desired Theta at End Time'
    )

    # Improve visibility
    ax.view_init(elev=20, azim=-135)  # Adjust 3D perspective

    plt.savefig(f"{loss_function}.png", dpi=600)
    #plt.show()
    print(f"Saved plot as '{loss_function}.png'.")
