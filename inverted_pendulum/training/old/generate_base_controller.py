import torch
import numpy as np

from PendulumController import PendulumController

device = torch.device("cpu")
controller = PendulumController().to(device)

# Use a previously generated random seed
random_seed = 4529

# Set the seeds for reproducibility
torch.manual_seed(random_seed)
np.random.seed(random_seed)

controller = PendulumController().to(device)
model_file = "controller_base.pth"
torch.save(controller.state_dict(), model_file)