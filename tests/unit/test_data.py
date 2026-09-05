"""Unit tests for the synthetic copy-task data pipeline."""

from __future__ import annotations

import pytest
import torch

from reconstruct_transformer.data import (
    BOS_IDX,
    EOS_IDX,
    PAD_IDX,
    CopyTaskDataset,
    collate_copy_examples,
    generate_copy_example,
    make_copy_dataloader,
)


def test_content_tokens_within_vocab_range() -> None:
    example = generate_copy_example(length=5, vocab_size=10)

    assert torch.all(example.source >= EOS_IDX + 1)
    assert torch.all(example.source < 10)


def test_target_has_bos_prefix_and_eos_suffix() -> None:
    example = generate_copy_example(length=4, vocab_size=10)

    assert example.target.shape == (6,)
    assert example.target[0] == BOS_IDX
    assert example.target[-1] == EOS_IDX
    assert torch.equal(example.target[1:-1], example.source)


def test_decoder_shift_relationship() -> None:
    example = generate_copy_example(length=3, vocab_size=10)

    assert torch.equal(example.decoder_input, example.target[:-1])
    assert torch.equal(example.labels, example.target[1:])
    assert example.decoder_input[0] == BOS_IDX
    assert example.labels[-1] == EOS_IDX


def test_rejects_invalid_length() -> None:
    with pytest.raises(ValueError, match="length"):
        generate_copy_example(length=0, vocab_size=10)


def test_rejects_too_small_vocab_size() -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        generate_copy_example(length=3, vocab_size=3)


def test_dataset_reproducible_with_seed() -> None:
    kwargs = dict(size=16, min_len=2, max_len=6, vocab_size=12, seed=42)
    first = CopyTaskDataset(**kwargs)
    second = CopyTaskDataset(**kwargs)

    assert first.lengths == second.lengths
    for index in range(len(first)):
        a = first[index]
        b = second[index]
        assert torch.equal(a.source, b.source)
        assert torch.equal(a.target, b.target)
        assert torch.equal(a.decoder_input, b.decoder_input)
        assert torch.equal(a.labels, b.labels)


def test_dataset_different_seeds_differ() -> None:
    first = CopyTaskDataset(size=32, min_len=2, max_len=8, vocab_size=20, seed=1)
    second = CopyTaskDataset(size=32, min_len=2, max_len=8, vocab_size=20, seed=2)

    assert any(
        not torch.equal(first[i].source, second[i].source)
        for i in range(len(first))
    )


def test_dataset_supports_variable_lengths() -> None:
    dataset = CopyTaskDataset(
        size=32, min_len=2, max_len=8, vocab_size=12, seed=0
    )

    assert len(set(dataset.lengths)) > 1


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(size=0, min_len=2, max_len=4, vocab_size=10),
        dict(size=8, min_len=0, max_len=4, vocab_size=10),
        dict(size=8, min_len=4, max_len=2, vocab_size=10),
        dict(size=8, min_len=2, max_len=4, vocab_size=3),
    ],
)
def test_dataset_rejects_invalid_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        CopyTaskDataset(**kwargs)


def test_collate_pads_to_longest_in_batch() -> None:
    examples = [
        generate_copy_example(2, 10),
        generate_copy_example(5, 10),
    ]

    batch = collate_copy_examples(examples)

    assert batch["source"].shape == (2, 5)
    assert batch["decoder_input"].shape == (2, 6)
    assert batch["labels"].shape == (2, 6)
    assert torch.equal(batch["source"][0, 2:], torch.full((3,), PAD_IDX))
    assert batch["source"][0, 0] == examples[0].source[0]


def test_dataloader_iteration() -> None:
    dataset = CopyTaskDataset(size=8, min_len=2, max_len=6, vocab_size=10, seed=0)
    loader = make_copy_dataloader(dataset, batch_size=3)

    batches = list(loader)

    assert len(batches) == 3
    for batch in batches:
        batch_size = batch["source"].shape[0]
        assert batch["source"].shape[0] == batch_size
        assert batch["decoder_input"].shape[0] == batch_size
        assert batch["labels"].shape[0] == batch_size
        assert batch["decoder_input"].shape[1] == batch["source"].shape[1] + 1
        assert torch.all(batch["source"] < 10)
        assert torch.all(batch["decoder_input"] < 10)
        assert bool(batch["decoder_input"][:, 0].eq(BOS_IDX).all())
