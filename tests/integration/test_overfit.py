"""Integration test proving the model overfits the copy task on CPU."""

from __future__ import annotations

import math

from reconstruct_transformer.training import TrainConfig, train


def test_copy_task_overfits_on_cpu(tmp_path) -> None:
    # Dropout is disabled so the tiny model can overfit cleanly; the Noam
    # warmup schedule is what lets it learn positional alignment at all.
    config = TrainConfig(
        seed=0,
        vocab_size=6,
        min_len=2,
        max_len=3,
        train_size=128,
        val_size=32,
        batch_size=32,
        d_model=32,
        num_heads=4,
        ffn_hidden=64,
        num_layers=2,
        dropout=0.0,
        model_max_len=16,
        learning_rate=1e-3,
        warmup_steps=100,
        epochs=300,
        device='cpu',
        output_dir=str(tmp_path),
    )

    result = train(config)

    initial_val_loss = result['history'][0]['val_loss']
    assert math.isfinite(initial_val_loss)
    assert result['final_val_accuracy'] >= 0.95
    assert result['final_val_loss'] < 0.5 * initial_val_loss
    assert all(math.isfinite(step['val_loss']) for step in result['history'])

    for key in (
        'final_train_loss',
        'final_val_loss',
        'final_val_accuracy',
        'best_epoch',
        'elapsed_seconds',
        'history',
    ):
        assert key in result
