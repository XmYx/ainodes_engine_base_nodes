import platform

try:
    import gc
    import torch

    def torch_gc():
        """Performs garbage collection for both Python and PyTorch CUDA tensors.

        This function collects Python garbage and clears the PyTorch CUDA cache
        and IPC (Inter-Process Communication) resources.
        """
        gc.collect()  # Collect Python garbage
        torch.cuda.empty_cache()  # Clear PyTorch CUDA cache
        torch.cuda.ipc_collect()  # Clear PyTorch CUDA IPC resources

except:

    def torch_gc():
        """Dummy function when torch is not available.

        This function does nothing and serves as a placeholder when torch is
        not available, allowing the rest of the code to run without errors.
        """
        pass


def get_torch_device():
    if "darwin" in platform.platform():
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        if torch.cuda.is_available():
            return torch.device(torch.cuda.current_device())
        else:
            return torch.device("cpu")
