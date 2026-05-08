import torch


def math_sinc(x: torch.Tensor) -> torch.Tensor:
    """MATLAB sinc: sin(pi*x) / (pi*x) == torch.sinc(x)."""
    return torch.sinc(x)
