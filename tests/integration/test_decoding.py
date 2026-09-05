"""Integration tests for autoregressive decoding, evaluation, and checkpoints."""

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch
from torch import nn

from reconstruct_transformer.data import (
    BOS_IDX,
    EOS_IDX,
    PAD_IDX,
    CopyTaskDataset,
    make_copy_dataloader,
)
from reconstruct_transformer.training import (
    TrainConfig,
    build_model,
    evaluate,
    greedy_decode,
    load_checkpoint,
    load_model_from_checkpoint,
    save_checkpoint,
)


class _ConstantModel(nn.Module):
    """Return logits that always predict ``token`` at every position."""

    def __init__(self, token: int, vocab_size: int = 6) -> None:
        super().__init__()
        self.token = token
        self.vocab_size = vocab_size

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        batch, seq_len = tgt.size(0), tgt.size(1)
        logits = torch.zeros(batch, seq_len, self.vocab_size)
        logits[..., self.token] = 1.0
        return logits


def test_greedy_decode_is_deterministic() -> None:
    model = _ConstantModel(token=3)
    src = torch.tensor([[3, 4]])

    first = greedy_decode(model, src, max_len=4)
    second = greedy_decode(model, src, max_len=4)

    assert torch.equal(first, second)


def test_greedy_decode_stops_at_eos() -> None:
    model = _ConstantModel(token=EOS_IDX)
    src = torch.tensor([[3, 4]])

    prediction = greedy_decode(model, src, max_len=10)

    assert prediction.tolist() == [EOS_IDX]


def test_greedy_decode_stops_at_max_len() -> None:
    model = _ConstantModel(token=3)  # never predicts EOS
    src = torch.tensor([[3, 4]])

    prediction = greedy_decode(model, src, max_len=5)

    assert prediction.tolist() == [3, 3, 3, 3, 3]


def test_greedy_decode_only_feeds_growing_prefix() -> None:
    class ProbeModel(_ConstantModel):
        def __init__(self) -> None:
            super().__init__(token=3)
            self.seen: list[torch.Tensor] = []

        def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
            self.seen.append(tgt.clone())
            return super().forward(src, tgt)

    model = ProbeModel()
    src = torch.tensor([[3, 4]])

    greedy_decode(model, src, max_len=4)

    expected = [
        [BOS_IDX],
        [BOS_IDX, 3],
        [BOS_IDX, 3, 3],
        [BOS_IDX, 3, 3, 3],
    ]
    assert [seen[0].tolist() for seen in model.seen] == expected


def test_checkpoint_round_trip(tmp_path) -> None:
    config = TrainConfig(
        vocab_size=8,
        d_model=16,
        num_heads=2,
        ffn_hidden=32,
        num_layers=1,
        dropout=0.0,
        model_max_len=16,
    )
    torch.manual_seed(0)
    first = build_model(config)
    first_optimizer = torch.optim.Adam(first.parameters(), lr=1e-3)
    metrics = {'val_accuracy': 0.9, 'val_loss': 0.1}
    path = tmp_path / 'ckpt.pt'
    save_checkpoint(path, first, first_optimizer, config, epoch=5, metrics=metrics)

    torch.manual_seed(123)
    second = build_model(config)
    second_optimizer = torch.optim.Adam(second.parameters(), lr=1e-3)
    ckpt_config, epoch, loaded_metrics = load_checkpoint(
        path, second, second_optimizer
    )

    assert epoch == 5
    assert loaded_metrics == metrics
    assert ckpt_config == asdict(config)

    src = torch.randint(3, 8, (2, 4))
    tgt = torch.randint(1, 8, (2, 5))
    torch.testing.assert_close(first(src, tgt), second(src, tgt))


def test_load_model_from_checkpoint(tmp_path) -> None:
    config = TrainConfig(
        vocab_size=8,
        d_model=16,
        num_heads=2,
        ffn_hidden=32,
        num_layers=1,
        dropout=0.0,
        model_max_len=16,
    )
    torch.manual_seed(0)
    first = build_model(config)
    first_optimizer = torch.optim.Adam(first.parameters(), lr=1e-3)
    save_checkpoint(
        tmp_path / 'ckpt.pt', first, first_optimizer, config, epoch=3,
        metrics={'val_accuracy': 0.7},
    )

    second, loaded_config, epoch, metrics = load_model_from_checkpoint(
        tmp_path / 'ckpt.pt'
    )

    assert epoch == 3
    assert metrics == {'val_accuracy': 0.7}
    assert loaded_config == config

    src = torch.randint(3, 8, (2, 4))
    tgt = torch.randint(1, 8, (2, 5))
    torch.testing.assert_close(first(src, tgt), second(src, tgt))


def test_evaluate_scores_perfect_copy_as_one() -> None:
    class PerfectCopyModel(nn.Module):
        def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
            batch, seq_len = tgt.size(0), tgt.size(1)
            logits = torch.full((batch, seq_len, 6), -10.0)
            source_len = int((src[0] != PAD_IDX).sum().item())
            for position in range(seq_len):
                if position < source_len:
                    logits[0, position, src[0, position]] = 10.0
                else:
                    logits[0, position, EOS_IDX] = 10.0
            return logits

    dataset = CopyTaskDataset(size=16, min_len=2, max_len=4, vocab_size=6, seed=0)
    loader = make_copy_dataloader(dataset, batch_size=8, shuffle=False)

    result = evaluate(PerfectCopyModel(), loader, torch.device('cpu'))

    assert result['token_accuracy'] == 1.0
    assert result['exact_match_accuracy'] == 1.0


def test_evaluate_scores_never_eos_as_no_exact_matches() -> None:
    model = _ConstantModel(token=3)  # never predicts EOS
    dataset = CopyTaskDataset(size=8, min_len=2, max_len=3, vocab_size=6, seed=0)
    loader = make_copy_dataloader(dataset, batch_size=8, shuffle=False)

    result = evaluate(model, loader, torch.device('cpu'))

    assert result['exact_match_accuracy'] == 0.0


def test_greedy_decode_rejects_batch_inputs() -> None:
    model = _ConstantModel(token=3)
    with pytest.raises(ValueError, match='batch decoding'):
        greedy_decode(model, torch.randint(1, 6, (2, 4)), max_len=4)


def test_decoding_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    config = TrainConfig(
        vocab_size=8,
        d_model=16,
        num_heads=2,
        ffn_hidden=32,
        num_layers=1,
        model_max_len=16,
    )
    model = build_model(config).to('cuda').eval()
    src = torch.randint(3, 8, (1, 4), device='cuda')

    prediction = greedy_decode(
        model, src, max_len=8, device=torch.device('cuda')
    )

    assert prediction.is_cuda
    assert torch.isfinite(model(src, torch.tensor([[BOS_IDX]], device='cuda'))).all()
