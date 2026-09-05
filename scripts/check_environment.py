"""Report and exercise the project's Python and PyTorch environment."""

from __future__ import annotations

import platform

import torch


def matrix_product(device: torch.device) -> torch.Tensor:
    """Return a deterministic matrix product on the requested device."""
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    right = torch.tensor([[2.0, 0.0], [1.0, 2.0]], device=device)
    return left @ right


def main() -> None:
    """Print environment metadata and run CPU/CUDA tensor operations."""
    cuda_available = torch.cuda.is_available()
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda or 'Not available'}")
    print(f"CUDA available: {cuda_available}")
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "Not available"
    print(f"GPU name: {gpu_name}")
    print(f"CPU tensor result: {matrix_product(torch.device('cpu')).tolist()}")
    if cuda_available:
        gpu_result = matrix_product(torch.device("cuda"))
        torch.cuda.synchronize()
        print(f"GPU tensor result: {gpu_result.cpu().tolist()}")


if __name__ == "__main__":
    main()
