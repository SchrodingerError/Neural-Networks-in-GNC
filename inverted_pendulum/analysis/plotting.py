import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_epoch_evolution(epochs, theta_over_epochs, desired_theta, save_path, title, num_steps, dt):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    time_steps = np.arange(num_steps) * dt

    theta_values = np.concatenate(theta_over_epochs)
    theta_min = np.min(theta_values)
    theta_max = np.max(theta_values)

    desired_range_min = desired_theta - 1.5 * np.pi
    desired_range_max = desired_theta + 1.5 * np.pi
    desired_range_min = max(theta_min, desired_range_min)
    desired_range_max = min(theta_max, desired_range_max)

    for epoch, theta_vals in reversed(list(zip(epochs, theta_over_epochs))):
        masked_theta_vals = np.array(theta_vals)
        masked_theta_vals[(masked_theta_vals < desired_range_min) | (masked_theta_vals > desired_range_max)] = np.nan
        ax.plot([epoch] * len(time_steps), time_steps, masked_theta_vals)

    epochs_array = np.array([epoch for epoch, _ in zip(epochs, theta_over_epochs)])
    ax.plot(epochs_array, [time_steps.max()] * len(epochs_array), [desired_theta] * len(epochs_array),
            color='r', linestyle='--', linewidth=2, label='Desired Theta at End Time')

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time (s)")
    ax.set_zlabel("Theta (rad)")
    ax.set_zscale('symlog')
    ax.set_title(title)
    ax.set_zlim(desired_range_min, desired_range_max)
    ax.view_init(elev=20, azim=-135)

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot as '{save_path}'.")


def plot_theta_vs_epoch(all_results, condition_name, desired_theta, save_path, title, specific_theta_index=-1):
    """
    Plots the theta values at a specific time over epochs for different loss functions for a specific condition, and adds a horizontal line at desired theta.

    :param all_results: Dictionary with structure {loss_function: {condition_name: (epochs, theta_over_epochs)}}
    :param condition_name: The key for the specific condition to plot.
    :param desired_theta: The y-value at which to draw a horizontal line across the plot.
    :param save_path: Path to save the final plot.
    :param title: Title of the plot.
    :param specific_theta_index: The index of the theta value to plot. Default is -1 for the last theta.
    """
    fig, ax = plt.subplots(figsize=(10, 7))  # Correct usage of plt.subplots for creating a figure and an axes.
    
    if condition_name not in all_results[next(iter(all_results))]:
        print(f"No data available for condition '{condition_name}'. Exiting plot function.")
        return

    for loss_function, conditions in all_results.items():
        if condition_name in conditions:
            epochs, theta_over_epochs = conditions[condition_name]
            # Extract final theta values for each epoch
            final_thetas = [thetas[specific_theta_index] for thetas in theta_over_epochs if thetas]  # Ensuring thetas is not empty
            ax.plot(epochs, final_thetas, label=f"{loss_function}")

    # Add a horizontal line at the desired_theta
    ax.axhline(y=desired_theta, color='r', linestyle='--', linewidth=2, label='Desired Theta')

    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Final Theta (rad)')
    ax.legend()

    plt.yscale('symlog')
    
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")
