import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint

# Define the neural network controller
class PendulumController(nn.Module):
    def __init__(self):
        super(PendulumController, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)

# Define pendulum dynamics
class PendulumDynamics(nn.Module):
    def __init__(self, m=10, g=9.81, R=1.0):
        super(PendulumDynamics, self).__init__()
        self.m = m
        self.g = g
        self.R = R
        self.torque_fn = None  # Set later before calling odeint

    def set_torque_fn(self, torque_fn):
        """ Set the neural network-based torque function """
        self.torque_fn = torque_fn

    def forward(self, t, state):
        theta, omega = state[:, :, 0], state[:, :, 1]  # Extract theta and omega

        # Ensure torque is correctly shaped
        torque = self.torque_fn(state)  # Neural network predicts torque
        torque = torch.clamp(torque.squeeze(-1), -250, 250)  # Limit torque

        dtheta_dt = omega
        domega_dt = (self.g / self.R) * torch.sin(theta) + torque / (self.m * self.R**2)

        return torch.stack([dtheta_dt, domega_dt], dim=2)

# Loss function with angle wrapping
def loss_fn(state, target_theta, torques):
    theta = state[:, :, 0]  # Extract theta trajectory
    omega = state[:, :, 1]  # Extract omega trajectory

    # Wrap theta to be within [-π, π]
    theta_wrapped = ((theta + torch.pi) % (2 * torch.pi)) - torch.pi

    alpha = 10.0   # Heavier weight for theta
    beta = 0.1     # Lighter weight for omega
    gamma = 0.01   # Regularization weight for motor torque
    delta = 100.0  # Large penalty for exceeding torque limit

    # Compute summation of squared differences for wrapped theta
    loss_theta = alpha * torch.sum((theta_wrapped - target_theta) ** 2)

    # Add penalty for omega (average remains to avoid scaling issues)
    loss_omega = beta * torch.mean(omega ** 2)

    # Add penalty for excessive torque usage (sum-based)
    loss_torque = gamma * torch.sum(torques ** 2)

    # Add penalty for torque exceeding 250
    over_limit_penalty = delta * torch.sum((torques.abs() > 250) * (torques.abs() - 250) ** 2)

    # Combine all losses
    loss = loss_theta + loss_omega + loss_torque + over_limit_penalty
    return loss

# Define batch of initial conditions
initial_conditions = [
    (0.1, 0.0),   # Small angle, zero velocity
    (0.5, 0.0),   # Medium angle, zero velocity
    (1.0, 0.0),   # Large angle, zero velocity
    (1.57, 0.5),  
    (0, -6.28),    
]

# Convert initial conditions to tensors
batch_size = len(initial_conditions)
theta_0 = torch.tensor([[ic[0]] for ic in initial_conditions], dtype=torch.float32)  # Shape: (batch_size, 1)
omega_0 = torch.tensor([[ic[1]] for ic in initial_conditions], dtype=torch.float32)  # Shape: (batch_size, 1)
state_0 = torch.cat([theta_0, omega_0], dim=1)  # Shape: (batch_size, 2)

# Simulation parameters
T_initial = torch.zeros((batch_size, 1), dtype=torch.float32)  # Shape: (batch_size, 1)
t_span = torch.linspace(0, 10, 200)  # Simulate for 10 seconds
target_theta = torch.zeros((batch_size, 1), dtype=torch.float32)  # Upright position

# Define the controller and optimizer
controller = PendulumController()
optimizer = optim.Adam(controller.parameters(), lr=0.01)
pendulum = PendulumDynamics()

# Training loop
num_epochs = 10_000
losses = []

for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Define torque function based on the neural network
    def torque_fn(state):
        # Ensure theta and omega have shape (batch_size, time_steps, 1)
        theta = state[:, :, 0].unsqueeze(-1)  
        omega = state[:, :, 1].unsqueeze(-1)  

        # Expand T_initial to match (batch_size, time_steps, 1)
        T_initial_expanded = T_initial.unsqueeze(1).expand(-1, theta.shape[1], -1)

        # Compute theta_ddot and ensure correct shape
        theta_ddot = ((pendulum.g / pendulum.R) * torch.sin(theta) + T_initial_expanded / (pendulum.m * pendulum.R**2))
        #theta_ddot = theta_ddot.unsqueeze(-1)  # 🔥 Remove extra dimension, now (batch_size, time_steps, 1)

        # 🔥 Ensure correct concatenation
        inputs = torch.cat([theta, omega, theta_ddot, T_initial_expanded], dim=2)  # Shape: (batch_size, time_steps, 4)

        # Pass through controller (neural network) and apply torque limit
        torque = controller(inputs)  # Predicted torque
        torque = torch.clamp(torque, -250, 250)  # Limit torque

        return torque




    # Set the torque function in the pendulum class
    pendulum.set_torque_fn(torque_fn)

    # Solve the forward dynamics for the **entire batch** at once
    state_traj = odeint(pendulum, state_0.unsqueeze(1).expand(-1, t_span.shape[0], -1), t_span, method='rk4')

    # Compute torques
    torques = torque_fn(state_traj)  # Shape: (batch_size, time_steps, 1)

    # Compute the loss over all initial conditions
    loss = loss_fn(state_traj, target_theta, torques)

    # Backpropagation and optimization
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    # Print loss every 50 epochs
    if epoch % 50 == 0:
        print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}")

# Save the trained model
torch.save(controller.state_dict(), "controller_batch_training.pth")
print("Trained model saved as 'controller_batch_training.pth'.")