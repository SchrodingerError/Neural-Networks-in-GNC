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
        clipped_theta_vals = np.clip(theta_vals, desired_range_min, desired_range_max)
        ax.plot([epoch] * len(time_steps), time_steps, clipped_theta_vals)

    epochs_array = np.array([epoch for epoch, _ in zip(epochs, theta_over_epochs)])
    ax.plot(epochs_array, [time_steps.max()] * len(epochs_array), [desired_theta] * len(epochs_array),
            color='r', linestyle='--', linewidth=2, label='Desired Theta at End Time')

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time (s)")
    ax.set_zlabel("Theta (rad)")
    ax.set_title(title)
    ax.set_zlim(desired_range_min, desired_range_max)
    ax.view_init(elev=20, azim=-135)

    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot as '{save_path}'.")

def plot_final_theta_vs_epoch(epochs, final_thetas, loss_functions, save_path):
    plt.figure()
    for final_theta, label in zip(final_thetas, loss_functions):
        plt.plot(epochs, final_theta, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Final Theta (rad)")
    plt.legend()
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    plt.savefig(save_path)
    plt.close()
