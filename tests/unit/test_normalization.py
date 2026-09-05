"""Unit tests for the custom LayerNorm implementation."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from reconstruct_transformer.normalization import LayerNorm


def test_preserves_input_shape() -> None:
    layer = LayerNorm(8)
    inputs = torch.randn(2, 4, 8)

    output = layer(inputs)

    assert output.shape == inputs.shape


def test_normalized_values_have_zero_mean_and_unit_variance() -> None:
    layer = LayerNorm(16, eps=1e-8)
    inputs = torch.randn(3, 5, 16) * 3.0 + 4.0

    output = layer(inputs)

    torch.testing.assert_close(
        output.mean(dim=-1), torch.zeros(3, 5), atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        output.var(dim=-1, unbiased=False),
        torch.ones(3, 5),
        atol=1e-5,
        rtol=1e-5,
    )


def test_matches_torch_layer_norm_reference() -> None:
    custom = LayerNorm(8, eps=1e-5)
    reference = nn.LayerNorm(8, eps=1e-5)
    inputs = torch.randn(2, 3, 8)
    with torch.no_grad():
        custom.weight.copy_(torch.linspace(0.5, 1.5, 8))
        custom.bias.copy_(torch.linspace(-0.4, 0.4, 8))
        reference.weight.copy_(custom.weight)
        reference.bias.copy_(custom.bias)

    actual = custom(inputs)
    expected = reference(inputs)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_custom_weight_and_bias_are_applied() -> None:
    layer = LayerNorm(3, eps=1e-5)
    inputs = torch.tensor([[1.0, 2.0, 3.0]])
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([2.0, 3.0, 4.0]))
        layer.bias.copy_(torch.tensor([-1.0, 0.5, 2.0]))

    mean = inputs.mean(dim=-1, keepdim=True)
    variance = inputs.var(dim=-1, unbiased=False, keepdim=True)
    expected = (inputs - mean) / torch.sqrt(variance + layer.eps)
    expected = expected * layer.weight + layer.bias

    torch.testing.assert_close(layer(inputs), expected)


@pytest.mark.parametrize("shape", [(7,), (2, 7), (2, 3, 4, 7)])
def test_supports_arbitrary_leading_dimensions(shape: tuple[int, ...]) -> None:
    layer = LayerNorm(7)
    inputs = torch.randn(shape)

    output = layer(inputs)

    assert output.shape == shape


def test_constant_input_is_finite_and_equals_bias() -> None:
    layer = LayerNorm(6)
    inputs = torch.full((2, 3, 6), 4.0)
    with torch.no_grad():
        layer.bias.copy_(torch.linspace(-1.0, 1.0, 6))

    output = layer(inputs)

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, layer.bias.expand_as(output))


def test_input_and_parameters_receive_finite_gradients() -> None:
    layer = LayerNorm(8)
    inputs = torch.randn(2, 3, 8, requires_grad=True)

    layer(inputs).square().sum().backward()

    assert inputs.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert torch.isfinite(layer.weight.grad).all()
    assert torch.isfinite(layer.bias.grad).all()


def test_preserves_float32_dtype() -> None:
    layer = LayerNorm(8)
    inputs = torch.randn(2, 3, 8, dtype=torch.float32)

    output = layer(inputs)

    assert output.dtype == torch.float32
    assert layer.weight.dtype == torch.float32
    assert layer.bias.dtype == torch.float32


def test_runs_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    layer = LayerNorm(8).to("cuda")
    inputs = torch.randn(2, 3, 8, device="cuda", requires_grad=True)

    output = layer(inputs)
    output.square().sum().backward()

    assert output.is_cuda
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


@pytest.mark.parametrize("normalized_shape", [0, -1])
def test_rejects_invalid_normalized_shape(normalized_shape: int) -> None:
    with pytest.raises(ValueError, match="normalized_shape"):
        LayerNorm(normalized_shape)


@pytest.mark.parametrize("eps", [0.0, -1e-5])
def test_rejects_non_positive_epsilon(eps: float) -> None:
    with pytest.raises(ValueError, match="eps"):
        LayerNorm(8, eps=eps)


def test_rejects_wrong_final_dimension() -> None:
    layer = LayerNorm(8)

    with pytest.raises(ValueError, match="expected final dimension 8"):
        layer(torch.randn(2, 3, 7))


def test_rejects_scalar_input() -> None:
    layer = LayerNorm(1)

    with pytest.raises(ValueError, match="expected final dimension"):
        layer(torch.tensor(1.0))
