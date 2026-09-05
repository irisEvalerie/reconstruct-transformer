"""Integration tests for the full encoder-decoder Transformer."""

from __future__ import annotations

import inspect

import pytest
import torch

import reconstruct_transformer.transformer as transformer_module
from reconstruct_transformer.transformer import Transformer


def make_model(**overrides: object) -> Transformer:
    kwargs: dict[str, object] = dict(
        src_vocab_size=20,
        tgt_vocab_size=20,
        d_model=8,
        num_heads=2,
        ffn_hidden=16,
        num_layers=2,
        src_pad_idx=0,
        tgt_pad_idx=0,
        max_len=16,
        dropout=0.0,
    )
    kwargs.update(overrides)
    return Transformer(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("batch", "src_len", "tgt_len"),
    [(1, 3, 2), (2, 5, 4), (3, 7, 6), (2, 1, 1)],
)
def test_logits_shape(batch: int, src_len: int, tgt_len: int) -> None:
    model = make_model().eval()
    src = torch.randint(1, 20, (batch, src_len))
    tgt = torch.randint(1, 20, (batch, tgt_len))

    logits = model(src, tgt)

    assert logits.shape == (batch, tgt_len, 20)


def test_causal_future_tokens_do_not_affect_earlier_logits() -> None:
    torch.manual_seed(0)
    model = make_model().eval()
    src = torch.randint(1, 20, (1, 6))
    tgt = torch.randint(1, 20, (1, 5))

    before = model(src, tgt)

    perturbed = tgt.clone()
    perturbed[:, -1] = (perturbed[:, -1] + 1) % 20
    after = model(src, perturbed)

    torch.testing.assert_close(before[:, :-1], after[:, :-1])
    assert not torch.allclose(before[:, -1], after[:, -1])


def test_source_padding_is_masked() -> None:
    torch.manual_seed(0)
    model = make_model().eval()
    src_short = torch.tensor([[3, 5, 2]])
    src_long = torch.tensor([[3, 5, 2, 0, 0]])
    tgt = torch.tensor([[1, 4, 6]])

    logits_short = model(src_short, tgt)
    logits_long = model(src_long, tgt)

    torch.testing.assert_close(logits_short, logits_long)


def test_forward_backward_and_finite_gradients() -> None:
    torch.manual_seed(0)
    model = make_model()
    src = torch.randint(1, 20, (2, 6))
    tgt = torch.randint(1, 20, (2, 5))

    logits = model(src, tgt)
    logits.sum().backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def test_reproducible_with_fixed_seed() -> None:
    def build() -> Transformer:
        torch.manual_seed(42)
        return make_model().eval()

    first, second = build(), build()
    src = torch.randint(1, 20, (2, 5))
    tgt = torch.randint(1, 20, (2, 4))

    torch.testing.assert_close(first(src, tgt), second(src, tgt))


def test_state_dict_round_trip_preserves_output() -> None:
    torch.manual_seed(0)
    first = make_model().eval()
    src = torch.randint(1, 20, (2, 5))
    tgt = torch.randint(1, 20, (2, 4))
    expected = first(src, tgt)

    second = make_model().eval()
    second.load_state_dict(first.state_dict())

    torch.testing.assert_close(second(src, tgt), expected)


def test_small_model_parameter_count() -> None:
    model = make_model()

    assert model.count_parameters() == sum(p.numel() for p in model.parameters())
    assert model.count_parameters() == 3508


def test_runs_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    model = make_model().to("cuda").eval()
    src = torch.randint(1, 20, (2, 5), device="cuda")
    tgt = torch.randint(1, 20, (2, 4), device="cuda")

    logits = model(src, tgt)

    assert logits.is_cuda
    assert torch.isfinite(logits).all()


def test_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="divisible"):
        make_model(d_model=8, num_heads=3)
    with pytest.raises(ValueError, match="num_layers"):
        make_model(num_layers=0)
    with pytest.raises(ValueError, match="src_pad_idx"):
        make_model(src_vocab_size=20, src_pad_idx=20)


def test_does_not_use_forbidden_transformer_apis() -> None:
    source = inspect.getsource(transformer_module)
    assert "nn.Transformer" not in source
    assert "nn.MultiheadAttention" not in source
    assert "F.scaled_dot_product_attention" not in source
    assert "functional.scaled_dot_product_attention" not in source
