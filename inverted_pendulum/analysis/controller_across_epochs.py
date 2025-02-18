import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool, cpu_count

# Define PendulumController class
from PendulumController import PendulumController

# Constants
g = 9.81  # Gravity
R = 1.0   # Length of the pendulum
m = 10.0  # Mass
dt = 0.02  # Time step
num_steps = 500  # Simulation time steps

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

def run_simulation(params):
    controller_file, initial_condition = params
    theta0, omega0, alpha0, desired_theta = initial_condition
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

# Named initial conditions
initial_conditions = {
    "small_perturbation": (0.1*np.pi, 0.0, 0.0, 0.0),
    "large_perturbation": (-np.pi, 0.0, 0.0, 0),
    "overshoot_vertical_test": (-0.1*np.pi, 2*np.pi, 0.0, 0.0),
    "overshoot_angle_test": (0.2*np.pi, 2*np.pi, 0.0, 0.3*np.pi),
    "extreme_perturbation": (4*np.pi, 0.0, 0.0, 0),
}

# Loss functions to iterate over
loss_functions = ["constant", "linear", "quadratic", "exponential", "inverse", "inverse_squared"]


epoch_start = 0   # Start of the epoch range
epoch_end = 1000  # End of the epoch range
epoch_step = 10    # Interval between epochs

if __name__ == "__main__":
    for condition_name, initial_condition in initial_conditions.items():
        full_path = f"/home/judson/Neural-Networks-in-GNC/inverted_pendulum/analysis/average_normalized/{condition_name}"
        os.makedirs(full_path, exist_ok=True)  # Create directory if it does not exist
        
        for loss_function in loss_functions:
            controller_dir = f"/home/judson/Neural-Networks-in-GNC/inverted_pendulum/training/normalized/average_normalized/{loss_function}/controllers"
            controller_files = sorted([f for f in os.listdir(controller_dir) if f.startswith("controller_") and f.endswith(".pth")])
            # Extract epoch numbers and filter based on the defined range and interval
            epoch_numbers = [int(f.split('_')[1].split('.')[0]) for f in controller_files]
            selected_epochs = [e for e in epoch_numbers if epoch_start <= e <= epoch_end and (e - epoch_start) % epoch_step == 0]

            # Filter the controller files to include only those within the selected epochs
            selected_controllers = [f for f in controller_files if int(f.split('_')[1].split('.')[0]) in selected_epochs]
            selected_controllers.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))

            # Setup parallel processing
            num_workers = min(cpu_count(), 16)  # Limit to 16 workers max
            print(f"Using {num_workers} parallel workers for {loss_function} with initial condition {condition_name}...")

            with Pool(processes=num_workers) as pool:
                params = [(controller_file, initial_condition) for controller_file in selected_controllers]
                results = pool.map(run_simulation, params)

            results.sort(key=lambda x: x[0])
            epochs, theta_over_epochs = zip(*results)

            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection='3d')
            time_steps = np.arange(num_steps) * dt

            # Plot the epochs in reverse order because we view it where epoch 0 is in front
            for epoch, theta_vals in reversed(list(zip(epochs, theta_over_epochs))):
                ax.plot([epoch] * len(time_steps), time_steps, theta_vals)


            # Add a horizontal line at desired_theta across all epochs and time steps
            epochs_array = np.array([epoch for epoch, _ in zip(epochs, theta_over_epochs)])
            desired_theta = initial_condition[-1]
            ax.plot(
                epochs_array,  # X-axis spanning all epochs
                [time_steps.max()] * len(epochs_array),  # Y-axis at the maximum time step
                [desired_theta] * len(epochs_array),  # Constant Z-axis value of desired_theta
                color='r', linestyle='--', linewidth=2, label='Desired Theta at End Time'
            )

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Time (s)")
            ax.set_zlabel("Theta (rad)")
            condition_text = f"IC_{'_'.join(map(lambda x: str(round(x, 2)), initial_condition))}"
            ax.set_title(f"Pendulum Angle Evolution for {loss_function} and {condition_text}")

            # Calculate the range of theta values across all epochs
            theta_values = np.concatenate(theta_over_epochs)
            theta_min = np.min(theta_values)
            theta_max = np.max(theta_values)

            # Determine the desired range around the desired_theta
            desired_range_min = desired_theta - 1 * np.pi
            desired_range_max = desired_theta + 1 * np.pi

            # Check if current theta values fall outside the desired range
            if theta_min < desired_range_min:
                desired_range_min = desired_range_min
            else:
                desired_range_min = theta_min

            if theta_max > desired_range_max:
                desired_range_max = desired_range_max
            else:
                desired_range_max = theta_max
            
            ax.set_zlim(desired_range_min, desired_range_max)

            ax.view_init(elev=20, azim=-135)  # Adjust 3D perspective

            plot_filename = os.path.join(full_path, f"{loss_function}.png")
            plt.savefig(plot_filename, dpi=300)
            plt.close()
            print(f"Saved plot as '{plot_filename}'.")