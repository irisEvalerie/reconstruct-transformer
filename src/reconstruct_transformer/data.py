'''Synthetic copy-task data pipeline.

Generates random token sequences and prepares the teacher-forced inputs and
labels the encoder-decoder Transformer consumes. No external data is
downloaded; all examples are produced locally from a controllable seed.
'''

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

# Special tokens. The first three ids are reserved; content tokens occupy the
# remaining ids from EOS_IDX + 1 up to vocab_size - 1.
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2


def _validate_vocab_size(vocab_size: int) -> None:
    if vocab_size <= EOS_IDX + 1:
        raise ValueError(
            'vocab_size must be greater than EOS_IDX + 1 so PAD, BOS, EOS and '
            'at least one content token all fit'
        )


@dataclass(frozen=True)
class CopyExample:
    '''One copy-task example.

    ``source`` holds the content tokens to copy (``(src_len,)``). ``target`` is
    ``[BOS] + source + [EOS]`` (``(tgt_len,)``). ``decoder_input`` and
    ``labels`` are the teacher-forced shift: ``decoder_input = target[:-1]`` and
    ``labels = target[1:]``, each of length ``tgt_len - 1``.
    '''

    source: torch.Tensor
    target: torch.Tensor
    decoder_input: torch.Tensor
    labels: torch.Tensor


def generate_copy_example(
    length: int,
    vocab_size: int,
    generator: torch.Generator | None = None,
) -> CopyExample:
    '''Build one example with ``length`` random content tokens.'''
    if length <= 0:
        raise ValueError('length must be greater than zero')
    _validate_vocab_size(vocab_size)

    content = torch.randint(
        low=EOS_IDX + 1,
        high=vocab_size,
        size=(length,),
        dtype=torch.long,
        generator=generator,
    )
    target = torch.cat(
        (torch.tensor([BOS_IDX]), content, torch.tensor([EOS_IDX]))
    )
    return CopyExample(
        source=content,
        target=target,
        decoder_input=target[:-1].clone(),
        labels=target[1:].clone(),
    )


class CopyTaskDataset(Dataset):
    '''Dataset of copy examples with lengths between ``min_len`` and ``max_len``.

    Randomness is fully controlled by ``seed``: sequence lengths and token
    contents are derived deterministically from the seed and example index, so
    identical seeds reproduce identical datasets regardless of access order.
    '''

    def __init__(
        self,
        size: int,
        min_len: int,
        max_len: int,
        vocab_size: int,
        seed: int = 0,
    ) -> None:
        if size <= 0:
            raise ValueError('size must be greater than zero')
        if min_len <= 0:
            raise ValueError('min_len must be greater than zero')
        if max_len < min_len:
            raise ValueError('max_len must be greater than or equal to min_len')
        _validate_vocab_size(vocab_size)

        self.size = size
        self.min_len = min_len
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.seed = seed

        lengths = torch.randint(
            low=min_len,
            high=max_len + 1,
            size=(size,),
            generator=torch.Generator().manual_seed(seed),
        )
        self.lengths = lengths.tolist()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> CopyExample:
        length = self.lengths[index]
        generator = torch.Generator().manual_seed(self.seed + index)
        return generate_copy_example(length, self.vocab_size, generator)


def collate_copy_examples(
    batch: list[CopyExample],
) -> dict[str, torch.Tensor]:
    '''Pad a batch of examples to the longest sequence in the batch.

    Returns tensors ``source``, ``decoder_input``, and ``labels`` of shape
    ``(batch, max_seq_len)``, padded with ``PAD_IDX``.
    '''
    source = pad_sequence(
        [example.source for example in batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )
    decoder_input = pad_sequence(
        [example.decoder_input for example in batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )
    labels = pad_sequence(
        [example.labels for example in batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )
    return {"source": source, "decoder_input": decoder_input, "labels": labels}


def make_copy_dataloader(
    dataset: CopyTaskDataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    '''Build a DataLoader around a copy-task dataset.'''
    generator = torch.Generator().manual_seed(dataset.seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_copy_examples,
        generator=generator,
    )
