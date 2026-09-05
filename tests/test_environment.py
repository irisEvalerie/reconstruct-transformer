"""Environment smoke tests for the reproducible tutorial baseline."""

from __future__ import annotations

import pytest
import torch

import reconstruct_transformer


def test_package_is_importable() -> None:
    assert reconstruct_transformer.__name__ == "reconstruct_transformer"


def test_torch_cpu_tensor_operation() -> None:
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    right = torch.tensor([[2.0, 0.0], [1.0, 2.0]])
    expected = torch.tensor([[4.0, 4.0], [10.0, 8.0]])
    assert torch.equal(left @ right, expected)


def test_torch_cuda_tensor_operation() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    values = torch.arange(4, dtype=torch.float32, device="cuda")
    result = (values.square() + 1).cpu()
    assert torch.equal(result, torch.tensor([1.0, 2.0, 5.0, 10.0]))
