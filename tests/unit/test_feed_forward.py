"""Unit tests for the position-wise feed-forward network."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from reconstruct_transformer.feed_forward import PositionwiseFeedForward


@pytest.mark.parametrize("shape", [(2, 4, 8), (1, 1, 8), (3, 7, 8), (2, 3, 4, 8)])
def test_preserves_shape_for_arbitrary_leading_dimensions(
    shape: tuple[int, ...],
) -> None:
    feed_forward = PositionwiseFeedForward(8, 16, drop_prob=0.0)

    output = feed_forward(torch.randn(shape))

    assert output.shape == shape


def test_matches_independent_manual_calculation() -> None:
    feed_forward = PositionwiseFeedForward(2, 3, drop_prob=0.0)
    inputs = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]])
    with torch.no_grad():
        feed_forward.linear1.weight.copy_(
            torch.tensor([[1.0, 2.0], [-1.0, 0.5], [2.0, -1.0]])
        )
        feed_forward.linear1.bias.copy_(torch.tensor([0.5, -0.5, 1.0]))
        feed_forward.linear2.weight.copy_(
            torch.tensor([[1.0, -2.0, 0.5], [0.25, 1.5, -1.0]])
        )
        feed_forward.linear2.bias.copy_(torch.tensor([-1.0, 2.0]))

    actual = feed_forward(inputs)
    hidden = inputs @ feed_forward.linear1.weight.T + feed_forward.linear1.bias
    activated = torch.maximum(hidden, torch.zeros_like(hidden))
    expected = activated @ feed_forward.linear2.weight.T + feed_forward.linear2.bias

    torch.testing.assert_close(actual, expected)


def test_changing_one_token_does_not_change_other_positions() -> None:
    feed_forward = PositionwiseFeedForward(4, 8, drop_prob=0.0)
    feed_forward.eval()
    original = torch.randn(2, 3, 4)
    changed = original.clone()
    changed[0, 1] += 100.0

    original_output = feed_forward(original)
    changed_output = feed_forward(changed)

    unchanged_positions = torch.ones(2, 3, dtype=torch.bool)
    unchanged_positions[0, 1] = False
    torch.testing.assert_close(
        original_output[unchanged_positions], changed_output[unchanged_positions]
    )


def test_identical_tokens_share_the_same_transformation() -> None:
    feed_forward = PositionwiseFeedForward(4, 8, drop_prob=0.0)
    token = torch.tensor([1.0, -2.0, 3.0, 0.5])
    inputs = token.expand(2, 5, -1).clone()

    output = feed_forward(inputs)

    torch.testing.assert_close(output, output[0, 0].expand_as(output))
    assert len(list(feed_forward.parameters())) == 4


def test_eval_mode_is_deterministic() -> None:
    feed_forward = PositionwiseFeedForward(8, 16, drop_prob=0.8)
    feed_forward.eval()
    inputs = torch.randn(2, 4, 8)

    first = feed_forward(inputs)
    second = feed_forward(inputs)

    torch.testing.assert_close(first, second)


def test_dropout_is_between_relu_and_second_linear_layer() -> None:
    feed_forward = PositionwiseFeedForward(2, 3, drop_prob=1.0)
    inputs = torch.ones(1, 2, 2)
    with torch.no_grad():
        feed_forward.linear1.weight.fill_(1.0)
        feed_forward.linear1.bias.fill_(1.0)
        feed_forward.linear2.weight.fill_(1.0)
        feed_forward.linear2.bias.copy_(torch.tensor([2.0, -3.0]))

    feed_forward.train()
    output = feed_forward(inputs)

    expected = feed_forward.linear2.bias.expand_as(output)
    torch.testing.assert_close(output, expected)


def test_train_mode_dropout_is_stochastic() -> None:
    torch.manual_seed(0)
    feed_forward = PositionwiseFeedForward(16, 64, drop_prob=0.5)
    feed_forward.train()
    inputs = torch.ones(8, 12, 16)

    first = feed_forward(inputs)
    second = feed_forward(inputs)

    assert not torch.equal(first, second)


def test_input_and_parameters_receive_finite_gradients() -> None:
    feed_forward = PositionwiseFeedForward(8, 16, drop_prob=0.0)
    inputs = torch.randn(2, 3, 8, requires_grad=True)

    feed_forward(inputs).square().mean().backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    for parameter in feed_forward.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_contains_exactly_two_linear_layers_and_relu() -> None:
    feed_forward = PositionwiseFeedForward(8, 16)
    linear_layers = [
        module for module in feed_forward.modules() if isinstance(module, nn.Linear)
    ]

    assert len(linear_layers) == 2
    assert isinstance(feed_forward.activation, nn.ReLU)
    assert isinstance(feed_forward.dropout, nn.Dropout)


def test_runs_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    feed_forward = PositionwiseFeedForward(8, 16).to("cuda")
    inputs = torch.randn(2, 3, 8, device="cuda", requires_grad=True)

    output = feed_forward(inputs)
    output.sum().backward()

    assert output.is_cuda
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((0, 16, 0.1), "d_model"),
        ((-1, 16, 0.1), "d_model"),
        ((8, 0, 0.1), "ffn_hidden"),
        ((8, -1, 0.1), "ffn_hidden"),
        ((8, 16, -0.1), "drop_prob"),
        ((8, 16, 1.1), "drop_prob"),
    ],
)
def test_rejects_invalid_constructor_arguments(
    arguments: tuple[int, int, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PositionwiseFeedForward(*arguments)


def test_rejects_wrong_final_dimension() -> None:
    feed_forward = PositionwiseFeedForward(8, 16)

    with pytest.raises(ValueError, match="expected final dimension 8"):
        feed_forward(torch.randn(2, 3, 7))


def test_rejects_scalar_input() -> None:
    feed_forward = PositionwiseFeedForward(1, 4)

    with pytest.raises(ValueError, match="expected final dimension"):
        feed_forward(torch.tensor(1.0))
