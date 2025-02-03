import torch

if torch.cuda.is_available():
    print("CUDA is available")
    print("Number of CUDA devices:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")