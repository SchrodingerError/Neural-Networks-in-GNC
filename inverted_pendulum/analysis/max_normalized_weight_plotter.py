import torch
import matplotlib.pyplot as plt

# Define time span
t_start, t_end, t_points = 0, 10, 1000
t_span = torch.linspace(t_start, t_end, t_points)

# Define weight functions
from time_weighting_functions import weight_functions

# Plot all weight functions
plt.figure(figsize=(10, 6))
for name, func in weight_functions.items():
    y_vals = func(t_span)
    print(f"{name}: {y_vals[0]:.3f} and {y_vals[-1]:.3f}")
    plt.plot(t_span, func(t_span), label=name)

plt.xlabel("Time (s)")
plt.ylabel("Weight Value")
plt.title("Max Normalized Weight Values")
plt.legend()
plt.grid(True)
plt.savefig("max_normalized_weights.png")