import numpy as np
import matplotlib.pyplot as plt

# Define time span
t_start, t_end, t_points = 0, 10, 1000
t_span = np.linspace(t_start, t_end, t_points)

# Define weight functions
weight_functions = {
    'constant': lambda t: np.ones_like(t),
    'linear': lambda t: (t+1) / (t+1).max(),
    'quadratic': lambda t: (t+1)**2 / ((t+1)**2).max(),
    'cubic': lambda t: (t+1)**3 / ((t+1)**3).max(),
    'inverse': lambda t: (t+1)**-1 / ((t+1)**-1).max(),
    'inverse_squared': lambda t: (t+1)**-2 / ((t+1)**-2).max(),
    'inverse_cubed': lambda t: (t+1)**-3 / ((t+1)**-3).max(),
    'linear_mirrored': lambda t: ((-t+10)+1) / ((-t+10)+1).max(),
    'quadratic_mirrored': lambda t: ((-t+10)+1)**2 / (((-t+10)+1)**2).max(),
    'cubic_mirrored': lambda t: ((-t+10)+1)**3 / (((-t+10)+1)**3).max(),
    'inverse_mirrored': lambda t: ((-t+10)+1)**-1 / (((-t+10)+1)**-1).max(),
    'inverse_squared_mirrored': lambda t: ((-t+10)+1)**-2 / (((-t+10)+1)**-2).max(),
    'inverse_cubed_mirrored': lambda t: ((-t+10)+1)**-3 / (((-t+10)+1)**-3).max()
}

# Plot all weight functions
plt.figure(figsize=(10, 6))
for name, func in weight_functions.items():
    plt.plot(t_span, func(t_span), label=name)

plt.xlabel("Time (s)")
plt.ylabel("Weight Value")
plt.title("Max Normalized Weight Values")
plt.legend()
plt.grid(True)
plt.savefig("max_normalized_weights.png")