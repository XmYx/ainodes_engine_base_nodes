from ainodes_frontend import singleton as gs
import torch, platform

def get_torch_device():
    if "macOS" in platform.platform():
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        if torch.cuda.is_available():
            return torch.device(torch.cuda.current_device())
        else:
            return torch.device("cpu")

gs.device = get_torch_device()