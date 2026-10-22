import torch

def create_tensor(method: str, shape: list, value: float = 0.0) -> torch.Tensor:
    """
    Returns a float32 tensor with the requested shape.
    """
    if method == 'ones':
        tensor = torch.ones(shape)
    elif method == 'zeros':
        tensor = torch.zeros(shape)
    else:
        tensor = torch.full(shape, value, dtype=torch.float32)

    return tensor
