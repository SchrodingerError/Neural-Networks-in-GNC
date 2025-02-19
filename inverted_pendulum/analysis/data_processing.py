import os

def get_controller_files(directory, epoch_range, epoch_step):
    controller_files = sorted([f for f in os.listdir(directory) if f.startswith("controller_") and f.endswith(".pth")])
    epoch_numbers = [int(f.split('_')[1].split('.')[0]) for f in controller_files]
    selected_epochs = [e for e in epoch_numbers if epoch_range[0] <= e <= epoch_range[1] and (e - epoch_range[0]) % epoch_step == 0]
    selected_controllers = [f for f in controller_files if int(f.split('_')[1].split('.')[0]) in selected_epochs]
    selected_controllers.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))
    return selected_controllers