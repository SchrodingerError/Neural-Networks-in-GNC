import torch
import torch.nn as nn
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
# 1) 3D Controller: [theta, omega, alpha] -> torque
# ----------------------------------------------------------------
class PendulumController3D(nn.Module):
    def __init__(self):
        super(PendulumController3D, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_3d):
        return self.net(x_3d)

# Load the trained 3D model
controller = PendulumController3D()
controller.load_state_dict(torch.load("controller_cpu_clamped_quadratic_time_penalty.pth"))
# controller.load_state_dict(torch.load("controller_cpu_clamped.pth"))
controller.eval()
print("3D Controller loaded.")

# ----------------------------------------------------------------
# 2) ODE: State = [theta, omega, alpha].
# ----------------------------------------------------------------
m = 10.0
g = 9.81
R = 1.0

def pendulum_ode_3d(t, state):
    theta, omega, alpha = state

    # Evaluate NN -> torque
    inp = torch.tensor([[theta, omega, alpha]], dtype=torch.float32)
    with torch.no_grad():
        torque = controller(inp).item()
        # Clamp torque to ±250 for consistency with training
        torque = np.clip(torque, -250, 250)

    alpha_des = (g/R)*np.sin(theta) + torque/(m*(R**2))

    dtheta = omega
    domega = alpha
    dalpha = alpha_des - alpha
    return [dtheta, domega, dalpha]

# ----------------------------------------------------------------
# 3) Validate for multiple initial conditions
# ----------------------------------------------------------------
initial_conditions_3d = [
    (0.1,  0.0, 0.0),
    (0.5,  0.0, 0.0),
    (1.0,  0.0, 0.0),
    (1.57, 0.5, 0.0),
    (0.0, -6.28, 0.0),
    (6.28, 6.28, 0.0),
]

t_span = (0, 20)
t_eval = np.linspace(0, 20, 2000)

for idx, (theta0, omega0, alpha0) in enumerate(initial_conditions_3d):
    sol = solve_ivp(
        pendulum_ode_3d,
        t_span,
        [theta0, omega0, alpha0],
        t_eval=t_eval,
        method='RK45'
    )

    t         = sol.t
    theta     = sol.y[0]
    omega     = sol.y[1]
    alpha_arr = sol.y[2]

    # Recompute torque over time
    torques = []
    alpha_des_vals = []
    for (th, om, al) in zip(theta, omega, alpha_arr):
        with torch.no_grad():
            torque_val = controller(torch.tensor([[th, om, al]], dtype=torch.float32)).item()
            torque_val = np.clip(torque_val, -250, 250)
        torques.append(torque_val)
        alpha_des_vals.append( (g/R)*np.sin(th) + torque_val/(m*(R**2)) )
    torques = np.array(torques)

    # Plot
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.plot(t, theta,     label="theta",      color="blue")
    ax1.plot(t, omega,     label="omega",      color="green")
    ax1.plot(t, alpha_arr, label="alpha",      color="red")
    # optional: ax1.plot(t, alpha_des_vals, label="alpha_des", color="red", linestyle="--")

    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("theta, omega, alpha")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(t, torques, label="torque", color="purple", linestyle="--")
    ax2.set_ylabel("Torque [Nm]")
    ax2.legend(loc="upper right")

    plt.title(f"IC (theta={theta0}, omega={omega0}, alpha={alpha0})")
    plt.tight_layout()
    plt.savefig(f"{idx+1}_validation.png")
    plt.close()
    print(f"Saved {idx+1}_validation.png")
