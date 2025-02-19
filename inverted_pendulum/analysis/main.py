from multiprocessing import Pool, cpu_count
import os
import numpy as np
from simulation import run_simulation
from data_processing import get_controller_files
from plotting import plot_3d_epoch_evolution

# Constants and setup
initial_conditions = {
    "small_perturbation": (0.1*np.pi, 0.0, 0.0, 0.0),
    "large_perturbation": (-np.pi, 0.0, 0.0, 0),
    "overshoot_vertical_test": (-0.1*np.pi, 2*np.pi, 0.0, 0.0),
    "overshoot_angle_test": (0.2*np.pi, 2*np.pi, 0.0, 0.3*np.pi),
    "extreme_perturbation": (4*np.pi, 0.0, 0.0, 0),
}
loss_functions = ["constant", "linear", "quadratic", "cubic", "inverse", "inverse_squared", "inverse_cubed"]
epoch_range = (0, 1000)  # Start and end of epoch range
epoch_step = 10          # Interval between epochs
dt = 0.02                # Time step for simulation
num_steps = 500          # Number of steps in each simulation

# Main execution
if __name__ == "__main__":
    for condition_name, initial_condition in initial_conditions.items():
        save_path_main = f"/home/judson/Neural-Networks-in-GNC/inverted_pendulum/analysis/average_normalized_new/{condition_name}"
        os.makedirs(save_path_main, exist_ok=True)  # Create directory if it does not exist
        for loss_function in loss_functions:
            # Construct the path to the controller directory
            directory = f"/home/judson/Neural-Networks-in-GNC/inverted_pendulum/training/average_normalized/{loss_function}/controllers"
            # Fetch the controller files according to the specified range and interval
            controllers = get_controller_files(directory, epoch_range, epoch_step)
            # Pack parameters for parallel processing
            tasks = [(c, initial_condition, directory, dt, num_steps) for c in controllers]
            
            # Execute simulations in parallel
            print("Starting worker processes")
            with Pool(min(cpu_count(), 16)) as pool:
                results = pool.map(run_simulation, tasks)

            # Sorting the results
            results.sort(key=lambda x: x[0])  # Assuming x[0] is the epoch number
            epochs, state_histories, torque_histories = zip(*results)  # Assuming results contain these

            # Convert state_histories to a more manageable form if necessary, e.g., just theta values
            theta_over_epochs = [[state[0] for state in history] for history in state_histories]

            # Plotting the 3D time evolution
            condition_text = f"IC_{'_'.join(map(lambda x: str(round(x, 2)), initial_condition))}"
            print(f"Plotting the 3d epoch evolution for {loss_function} under {condition_text}")
            title = f"Pendulum Angle Evolution for {loss_function} and {condition_text}"
            save_path = os.path.join(save_path_main, f"{loss_function}.png")
            desired_theta = initial_condition[-1]
            plot_3d_epoch_evolution(epochs, theta_over_epochs, desired_theta, save_path, title, num_steps, dt)

            print(f"Completed plotting for {loss_function} under {condition_name} condition.\n")
