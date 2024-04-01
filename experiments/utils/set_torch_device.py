import torch
import os

def set_torch_device():
    """
    Attempts to set device to CUDA/mps if available, CPU otherwise.
    """
    device_name = "CPU"
    if os.name == "posix":
        device = torch.device('mps') if torch.cuda.is_available() else torch.device('cpu')
    if os.name == "nt":
        if (torch.cuda.is_available() != True):
            device = torch.device('cpu')
        else:
            id = torch.cuda.current_device()
            device = torch.device('cuda')
            device_name = torch.cuda.get_device_name(id)

    return device, device_name