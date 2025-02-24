from multiprocessing import Pool, cpu_count
import os
import numpy as np
from simulation import run_simulation
from data_processing import get_controller_files
from plotting import plot_3d_epoch_evolution, plot_theta_vs_epoch

# Constants and setup
initial_conditions = {
    "small_perturbation": (0.1*np.pi, 0.0, 0.0, 0.0),
    "large_perturbation": (-np.pi, 0.0, 0.0, 0),
    "overshoot_vertical_test": (-0.1*np.pi, 2*np.pi, 0.0, 0.0),
    "overshoot_angle_test": (0.2*np.pi, 2*np.pi, 0.0, 0.3*np.pi),
    "extreme_perturbation": (4*np.pi, 0.0, 0.0, 0),
}
loss_functions = ["constant", "linear", "quadratic", "cubic", "inverse", "inverse_squared", "inverse_cubed"]
epoch_range = (0, 3)  # Start and end of epoch range
epoch_step = 1          # Interval between epochs
dt = 0.02                # Time step for simulation
num_steps = 500          # Number of steps in each simulation

# Main execution
if __name__ == "__main__":
    all_results = {}  # Dictionary to store results by loss function

    for condition_name, initial_condition in initial_conditions.items():
        condition_text = f"IC_{'_'.join(map(lambda x: str(round(x, 2)), initial_condition))}"
        desired_theta = initial_condition[-1]
        condition_path = f"/home/judson/Neural-Networks-in-GNC/inverted_pendulum/analysis/max_normalized/{condition_name}"
        os.makedirs(condition_path, exist_ok=True)  # Create directory if it does not exist
        for loss_function in loss_functions:
            # Construct the path to the controller directory
            directory = f"/home/judson/Neural-Networks-in-GNC/inverted_pendulum/training/max_normalized/{loss_function}/controllers"
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
            
            # Store results for later use
            if loss_function not in all_results:
                all_results[loss_function] = {}
            all_results[loss_function][condition_name] = (epochs, theta_over_epochs)

            # continue
            # Plotting the 3D epoch evolution
            print(f"Plotting the 3d epoch evolution for {loss_function} under {condition_text}")
            title = f"Pendulum Angle Evolution for {loss_function} and {condition_text}"
            save_path = os.path.join(condition_path, f"epoch_evolution")
            save_path = os.path.join(save_path, f"{loss_function}.png")
            plot_3d_epoch_evolution(epochs, theta_over_epochs, desired_theta, save_path, title, num_steps, dt)
            print("")

        # Plot the  theta as a function of epoch for all loss functions
        continue

        specific_theta_index = num_steps // 2
        save_path = os.path.join(condition_path, f"theta_at_5sec_across_epochs.png")
        plot_theta_vs_epoch(all_results, condition_name, desired_theta, save_path, f"Theta at 5 Seconds across Epochs for {condition_text}", specific_theta_index)

        specific_theta_index = -1
        save_path = os.path.join(condition_path, f"final_theta_across_epochs.png")
        plot_theta_vs_epoch(all_results, condition_name, desired_theta, save_path, f"Final Theta across Epochs for {condition_text}", specific_theta_index)

        print(f"Completed plotting for all loss functions under {condition_name} condition.\n")

    import json

    with open("all_results.json", 'w') as file:
        json.dump(all_results, file)