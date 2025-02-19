import os
import numpy as np
import torch
from PendulumController import PendulumController

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
        a = (9.81 / 1.0) * np.sin(th) + torque / (10.0 * 1.0**2)
        return np.array([om, a, 0])  # dtheta, domega, dalpha

    # RK4 integration
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
    new_state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    # Calculate the pseudo torque applied at the intervale based on the torques at each substep
    torque = (torque1 + 2 * torque2 + 2 * torque3 + torque4) / 6.0
    return new_state, torque

def run_simulation(params):
    controller_file, initial_condition, controller_dir, dt, num_steps = params
    controller_path = os.path.join(controller_dir, controller_file)
    controller = PendulumController()
    controller.load_state_dict(torch.load(controller_path))
    controller.eval()

    theta0, omega0, alpha0, desired_theta = initial_condition
    state = np.array([theta0, omega0, alpha0])
    state_history = []
    torque_history = []

    for _ in range(num_steps):
        state, torque = pendulum_ode_step(state, dt, desired_theta, controller)
        state_history.append(state)
        torque_history.append(torque)

    epoch = int(controller_file.split('_')[1].split('.')[0])

    return epoch, state_history, torque_history
