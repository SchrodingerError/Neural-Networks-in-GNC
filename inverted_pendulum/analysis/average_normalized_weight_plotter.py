import numpy as np
import matplotlib.pyplot as plt

# Define time span
t_start, t_end, t_points = 0, 10, 1000
t_span = np.linspace(t_start, t_end, t_points)

# Define normalized weight functions
weight_functions = {
    'constant': lambda t: np.ones_like(t) / np.ones_like(t).mean(),
    'linear': lambda t: ((t+1) / (t+1).max()) / ((t+1) / (t+1).max()).mean(),
    'quadratic': lambda t: ((t+1)**2 / ((t+1)**2).max()) / ((t+1)**2 / ((t+1)**2).max()).mean(),
    'cubic': lambda t: ((t+1)**3 / ((t+1)**3).max()) / ((t+1)**3 / ((t+1)**3).max()).mean(),
    'inverse': lambda t: ((t+1)**-1 / ((t+1)**-1).max()) / ((t+1)**-1 / ((t+1)**-1).max()).mean(),
    'inverse_squared': lambda t: ((t+1)**-2 / ((t+1)**-2).max()) / ((t+1)**-2 / ((t+1)**-2).max()).mean(),
    'inverse_cubed': lambda t: ((t+1)**-3 / ((t+1)**-3).max()) / ((t+1)**-3 / ((t+1)**-3).max()).mean()
}

# Plot all weight functions
plt.figure(figsize=(10, 6))
for name, func in weight_functions.items():
    plt.plot(t_span, func(t_span), label=name)

plt.xlabel("Time (s)")
plt.ylabel("Weight Value")
plt.title("Average Normalized Weight Values")
plt.legend()
plt.grid(True)
plt.savefig("average_normalized_weights.png")
