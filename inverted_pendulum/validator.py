import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load Controller
controller_file_name = "controller_with_desired_theta.pth"

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

controller = PendulumController()
controller.load_state_dict(torch.load(controller_file_name))
controller.eval()
print(f"{controller_file_name} loaded.")

# Constants
m = 10.0
g = 9.81
R = 1.0

# Time step settings
dt = 0.01  # Fixed time step
T = 20.0   # Total simulation time
num_steps = int(T / dt)
t_eval = np.linspace(0, T, num_steps)

# Define ODE System (RK4) - Also Returns Torque
def pendulum_ode_step(state, dt, desired_theta):
    theta, omega, alpha = state

    def compute_torque(th, om, al):
        # Evaluate NN -> torque
        inp = torch.tensor([[th, om, al, desired_theta]], dtype=torch.float32)
        with torch.no_grad():
            torque = controller(inp)
            torque = torch.clamp(torque, -250, 250)
        return float(torque)

    def derivatives(state, torque):
        th, om, al = state
        a = (g / R) * np.sin(th) + torque / (m * R**2)
        return np.array([om, a, 0])  # dtheta, domega, dalpha

    # Compute k1
    torque1 = compute_torque(theta, omega, alpha)
    k1 = dt * derivatives(state, torque1)

    # Compute k2
    state_k2 = state + 0.5 * k1
    torque2 = compute_torque(state_k2[0], state_k2[1], state_k2[2])
    k2 = dt * derivatives(state_k2, torque2)

    # Compute k3
    state_k3 = state + 0.5 * k2
    torque3 = compute_torque(state_k3[0], state_k3[1], state_k3[2])
    k3 = dt * derivatives(state_k3, torque3)

    # Compute k4
    state_k4 = state + k3
    torque4 = compute_torque(state_k4[0], state_k4[1], state_k4[2])
    k4 = dt * derivatives(state_k4, torque4)

    # Update state using RK4 formula
    new_state = state +  (k1 + 2*k2 + 2*k3 + k4) / 6.0

    # Compute weighted final torque
    final_torque = (torque1 + 2*torque2 + 2*torque3 + torque4) / 6.0

    return new_state, final_torque




# Run Simulations for Different Initial Conditions
# [theta0, omega0, alpha0, desired_theta]
in_sample_cases = [
    # Theta perturbations
    (1/6 * np.pi,    0.0, 0.0, 0.0),
    (-1/6 * np.pi,   0.0, 0.0, 0.0),
    (2/3 * np.pi,    0.0, 0.0, 0.0),
    (-2/3 * np.pi,   0.0, 0.0, 0.0),

    # Omega perturbations
    (0.0, 1/3 * np.pi,      0.0, 0.0),
    (0.0, -1/3 * np.pi,     0.0, 0.0),
    (0.0, 2 * np.pi,        0.0, 0.0),
    (0.0, -2 * np.pi,       0.0, 0.0),

    # Return to non-zero theta
    (0.0, 0.0, 0.0,     2*np.pi),
    (0.0, 0.0, 0.0,     -2*np.pi),
    (0.0, 0.0, 0.0,     1/2 * np.pi),
    (0.0, 0.0, 0.0,     -1/2 *np.pi),
    (0.0, 0.0, 0.0,     1/3 * np.pi),
    (0.0, 0.0, 0.0,     -1/3 *np.pi),

    # Mix cases
    (1/4 * np.pi,    1 * np.pi,     0.0,      0.0),
    (-1/4 * np.pi,   -1 * np.pi,    0.0,     0.0),
    (1/2 * np.pi,    -1 * np.pi,    0.0,     1/3 * np.pi),
    (-1/2 * np.pi,   1 * np.pi,     0.0,      -1/3 *np.pi),
    (1/4 * np.pi,    1 * np.pi,     0.0,      2 * np.pi),
    (-1/4 * np.pi,   -1 * np.pi,    0.0,     2 * np.pi),
    (1/2 * np.pi,    -1 * np.pi,    0.0,     4 * np.pi),
    (-1/2 * np.pi,   1 * np.pi,     0.0,      -4 *np.pi),
]

# Validation in-sample cases
print("Performing in-sample validation")

losses = []
final_thetas = []

for idx, (theta0, omega0, alpha0, desired_theta) in enumerate(in_sample_cases):
    state = np.array([theta0, omega0, alpha0])

    theta_vals, omega_vals, alpha_vals, torque_vals = [], [], [], []

    for _ in range(num_steps):
        # Save values
        theta_vals.append(state[0])
        omega_vals.append(state[1])
        alpha_vals.append(state[2])

        # Compute ODE step with real state
        state, torque = pendulum_ode_step(state, dt, desired_theta)

        # Store torque
        torque_vals.append(torque)

    # Convert lists to arrays
    theta_vals = np.array(theta_vals)
    omega_vals = np.array(omega_vals)
    alpha_vals = np.array(alpha_vals)
    torque_vals = np.array(torque_vals)

    # Get the final theta of the system at t=t_final
    final_theta = theta_vals[-1]
    final_thetas.append(final_theta)

    # Calculate this specific condition's loss
    loss = 1e3 * np.mean((theta_vals - desired_theta)**2)
    losses.append(loss)

    # Plot Results
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(t_eval, theta_vals, label="theta", color="blue")
    ax1.plot(t_eval, omega_vals, label="omega", color="green")
    ax1.plot(t_eval, alpha_vals, label="alpha", color="red")
    ax1.axhline(desired_theta, label="Desired Theta", color="black")

    # Draw horizontal lines at theta = 2n*pi (as many as fit within range)
    y_min = min(np.min(theta_vals), np.min(omega_vals), np.min(alpha_vals))
    y_max = max(np.max(theta_vals), np.max(omega_vals), np.max(alpha_vals))

    n_min = int(np.ceil(y_min / (2 * np.pi)))
    n_max = int(np.floor(y_max / (2 * np.pi)))
    theta_lines = [2 * n * np.pi for n in range(n_min, n_max + 1)]

    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("theta, omega, alpha")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(t_eval, torque_vals, label="torque", color="purple", linestyle="--")
    ax2.set_ylabel("Torque [Nm]")
    ax2.legend(loc="upper right")

    plt.title(f"IC (theta={theta0}, omega={omega0}, alpha={alpha0})")
    plt.tight_layout()

    filename = f"{idx+1}_theta0_{theta0:.3f}_omega0_{omega0:.3f}_alpha0_{alpha0:.3f}_desiredtheta_{desired_theta:.3f}_finaltheta_{final_theta:.3f}.png"
    plt.savefig(f"validation/in-sample/{filename}")
    plt.close()

    print(f"Saved in-sample validation case {idx+1}")

# Create a DataFrame for tabular representation
df_losses = pd.DataFrame(in_sample_cases, columns=["theta0", "omega0", "alpha0", "desired_theta"])
df_losses["final_theta"] = final_theta
df_losses["loss"] = losses

# Add run # column
df_losses.insert(0, "Run #", range(1, len(in_sample_cases) + 1))

# Print the table
print(df_losses.to_string(index=False))


# Out-of-sample validation
print("\nPerforming out-of-sample validation")

# Out of sample cases previously generated by numpy
out_sample_cases = [
    (-2.198958, -4.428501, 0.450833, 0.000000),
    (1.714196, -0.769896, 0.202738, 0.000000),
    (0.241195, -5.493715, 0.438996, 0.000000),
    (0.030605, 4.901513, -0.479243, 0.000000),
    (1.930445, -1.301926, -0.454050, 0.000000),
    (-0.676063, 4.246865, 0.036303, 0.000000),
    (0.734920, -5.925202, 0.047097, 0.000000),
    (-3.074471, -3.535424, 0.315438, 0.000000),
    (-0.094486, 6.111091, 0.150525, 0.000000),
    (-1.647671, 5.720526, 0.334181, 0.000000),
    (-2.611260, 5.087704, 0.045460, -3.610785),
    (1.654137, 0.982081, -0.192725, 1.003872),
    (-2.394899, 3.550547, -0.430938, 3.261897),
    (0.474917, 0.555166, -0.285173, 1.866752),
    (-0.640369, -4.678490, -0.340663, 3.150098),
    (1.747517, -3.248204, -0.001520, 1.221787),
    (2.505283, -2.875006, -0.065617, -3.690269),
    (1.337244, 2.221707, 0.044979, -2.459730),
    (1.531012, 2.230981, -0.291206, -1.924535),
    (-1.065792, 4.320740, 0.075405, -1.550644),
]



for idx, (theta0, omega0, alpha0, desired_theta) in enumerate(out_sample_cases):
    state = np.array([theta0, omega0, alpha0])

    theta_vals, omega_vals, alpha_vals, torque_vals = [], [], [], []

    for _ in range(num_steps):
        # Save values
        theta_vals.append(state[0])
        omega_vals.append(state[1])
        alpha_vals.append(state[2])

        # Compute ODE step with real state
        state, torque = pendulum_ode_step(state, dt, desired_theta)

        # Store torque
        torque_vals.append(torque)

    # Convert lists to arrays
    theta_vals = np.array(theta_vals)
    omega_vals = np.array(omega_vals)
    alpha_vals = np.array(alpha_vals)
    torque_vals = np.array(torque_vals)

    # Get the final theta of the system at t=t_final
    final_theta = theta_vals[-1]
    final_thetas.append(final_theta)

    # Calculate this specific condition's loss
    loss = 1e3 * np.mean((theta_vals - desired_theta)**2)
    losses.append(loss)

    # Plot Results
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(t_eval, theta_vals, label="theta", color="blue")
    ax1.plot(t_eval, omega_vals, label="omega", color="green")
    ax1.plot(t_eval, alpha_vals, label="alpha", color="red")
    ax1.axhline(desired_theta, label="Desired Theta", color="black")

    # Draw horizontal lines at theta = 2n*pi (as many as fit within range)
    y_min = min(np.min(theta_vals), np.min(omega_vals), np.min(alpha_vals))
    y_max = max(np.max(theta_vals), np.max(omega_vals), np.max(alpha_vals))

    n_min = int(np.ceil(y_min / (2 * np.pi)))
    n_max = int(np.floor(y_max / (2 * np.pi)))
    theta_lines = [2 * n * np.pi for n in range(n_min, n_max + 1)]

    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("theta, omega, alpha")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(t_eval, torque_vals, label="torque", color="purple", linestyle="--")
    ax2.set_ylabel("Torque [Nm]")
    ax2.legend(loc="upper right")

    plt.title(f"IC (theta={theta0}, omega={omega0}, alpha={alpha0})")
    plt.tight_layout()

    filename = f"{idx+1}_theta0_{theta0:.3f}_omega0_{omega0:.3f}_alpha0_{alpha0:.3f}_desiredtheta_{desired_theta:.3f}_finaltheta_{final_theta:.3f}.png"
    plt.savefig(f"validation/out-of-sample/{filename}")
    plt.close()

    print(f"Saved out-of-sample validation case {idx+1}")


# Create a DataFrame for tabular representation
df_losses = pd.DataFrame(out_sample_cases, columns=["theta0", "omega0", "alpha0", "desired_theta"])
df_losses["final_theta"] = final_theta
df_losses["loss"] = losses

# Add run # column
df_losses.insert(0, "Run #", range(1, len(out_sample_cases) + 1))

# Print the table
print(df_losses.to_string(index=False))