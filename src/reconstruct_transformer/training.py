'''Training, evaluation, and checkpointing for the copy task.

This module keeps the whole training story in one place: configuration, seeding,
device selection, model/dataloader construction, the standard teacher-forcing
training loop, and checkpoint save/load. Decoding is added in a later stage.
'''

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy
import torch
import yaml
from torch import nn

from .data import BOS_IDX, EOS_IDX, PAD_IDX, CopyTaskDataset, make_copy_dataloader
from .transformer import Transformer


@dataclass
class TrainConfig:
    '''Flat configuration for a copy-task training run.

    ``load_config`` flattens the nested ``configs/copy_task.yaml`` into this
    structure; ``steps`` is reserved for step-based runs and ignored by the
    epoch-based loop in :func:`train`.
    '''

    seed: int = 0
    vocab_size: int = 11
    min_len: int = 2
    max_len: int = 8
    train_size: int = 2048
    val_size: int = 256
    batch_size: int = 32
    d_model: int = 64
    num_heads: int = 4
    ffn_hidden: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    model_max_len: int = 32
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-9
    epochs: int = 20
    steps: int | None = None
    grad_clip_norm: float | None = 1.0
    warmup_steps: int = 0
    device: str = 'auto'
    output_dir: str = 'outputs/copy_task'


def load_config(path: str | Path) -> TrainConfig:
    '''Load a copy-task YAML config and flatten it into a :class:`TrainConfig`.'''
    with open(path, encoding='utf-8') as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError('config must be a YAML mapping')

    for key in (
        'seed',
        'vocab_size',
        'min_len',
        'max_len',
        'train_size',
        'val_size',
        'batch_size',
        'device',
        'output_dir',
    ):
        if key not in raw:
            raise ValueError(f'config is missing required key: {key}')

    model = raw.get('model') or {}
    optimizer = raw.get('optimizer') or {}
    training = raw.get('training') or {}

    if optimizer.get('name', 'adam') != 'adam':
        raise ValueError("only the 'adam' optimizer is supported")

    return TrainConfig(
        seed=raw['seed'],
        vocab_size=raw['vocab_size'],
        min_len=raw['min_len'],
        max_len=raw['max_len'],
        train_size=raw['train_size'],
        val_size=raw['val_size'],
        batch_size=raw['batch_size'],
        d_model=model.get('d_model', 64),
        num_heads=model.get('num_heads', 4),
        ffn_hidden=model.get('ffn_hidden', 128),
        num_layers=model.get('num_layers', 2),
        dropout=model.get('dropout', 0.1),
        model_max_len=model.get('max_len', 100),
        learning_rate=optimizer.get('learning_rate', 3e-4),
        beta1=optimizer.get('beta1', 0.9),
        beta2=optimizer.get('beta2', 0.98),
        eps=optimizer.get('eps', 1e-9),
        epochs=training.get('epochs', 20),
        steps=training.get('steps'),
        grad_clip_norm=training.get('grad_clip_norm', 1.0),
        warmup_steps=training.get('warmup_steps', 0),
        device=raw['device'],
        output_dir=raw['output_dir'],
    )


def seed_everything(seed: int) -> None:
    '''Seed Python, NumPy, and PyTorch for reproducible runs.

    The model uses only matmul, softmax, and layer normalization, so cuDNN
    determinism flags are deliberately not touched; they add no reproducibility
    benefit here and can slow down training on some GPUs.
    '''
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    '''Resolve a device string to a :class:`torch.device`.'''
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device in ('cuda', 'cpu'):
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available')
        return torch.device(device)
    raise ValueError(f"unknown device: {device!r}")


def build_model(config: TrainConfig) -> Transformer:
    '''Build the copy-task Transformer from a config.'''
    return Transformer(
        src_vocab_size=config.vocab_size,
        tgt_vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        ffn_hidden=config.ffn_hidden,
        num_layers=config.num_layers,
        src_pad_idx=PAD_IDX,
        tgt_pad_idx=PAD_IDX,
        max_len=config.model_max_len,
        dropout=config.dropout,
    )


def build_dataloaders(
    config: TrainConfig,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    '''Build shuffled train and unshuffled validation loaders.

    The validation seed is offset far from the training seed so the two sets
    draw independent sequences.
    '''
    train_dataset = CopyTaskDataset(
        size=config.train_size,
        min_len=config.min_len,
        max_len=config.max_len,
        vocab_size=config.vocab_size,
        seed=config.seed,
    )
    val_dataset = CopyTaskDataset(
        size=config.val_size,
        min_len=config.min_len,
        max_len=config.max_len,
        vocab_size=config.vocab_size,
        seed=config.seed + 1_000_000,
    )
    train_loader = make_copy_dataloader(
        train_dataset, config.batch_size, shuffle=True
    )
    val_loader = make_copy_dataloader(
        val_dataset, config.batch_size, shuffle=False
    )
    return train_loader, val_loader


def _build_noam_scheduler(
    optimizer: torch.optim.Optimizer,
    d_model: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    '''Return the paper's Noam warmup-then-decay schedule, or ``None``.

    The learning rate follows ``d_model^-0.5 * min(step^-0.5, step *
    warmup_steps^-1.5)``, matching "Attention Is All You Need".
    '''
    if not warmup_steps or warmup_steps <= 0:
        return None

    def lr_lambda(step: int) -> float:
        step = max(step, 1)
        return (d_model ** -0.5) * min(
            step ** -0.5, step * (warmup_steps ** -1.5)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _evaluate(
    model: Transformer,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    '''Return (average loss, token accuracy) over a loader, ignoring padding.'''
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            source = batch['source'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            labels = batch['labels'].to(device)

            logits = model(source, decoder_input)
            loss = criterion(logits.permute(0, 2, 1), labels)

            mask = labels != PAD_IDX
            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()
            predictions = logits.argmax(dim=-1)
            total_correct += ((predictions == labels) & mask).sum().item()

    average_loss = total_loss / total_tokens if total_tokens else 0.0
    accuracy = total_correct / total_tokens if total_tokens else 0.0
    return average_loss, accuracy


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    epoch: int,
    metrics: dict[str, Any],
) -> str:
    '''Save model, optimizer, config, epoch, and metrics to a single file.'''
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': asdict(config),
        'epoch': epoch,
        'metrics': metrics,
    }
    torch.save(checkpoint, path)
    return str(path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, Any], int, dict[str, Any]]:
    '''Restore a checkpoint, returning (config, epoch, metrics).'''
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['config'], checkpoint['epoch'], checkpoint['metrics']


def load_model_from_checkpoint(
    path: str | Path,
    device: torch.device | None = None,
) -> tuple[Transformer, TrainConfig, int, dict[str, Any]]:
    '''Rebuild a model and its config from a saved checkpoint.'''
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = TrainConfig(**checkpoint['config'])
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if device is not None:
        model = model.to(device)
    return model, config, checkpoint['epoch'], checkpoint['metrics']


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    max_len: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    '''Autoregressively decode a single source sequence with greedy search.

    Only a single example is supported: ``src`` must have shape
    ``(1, src_len)``. Generation starts from ``BOS_IDX``; at each step the model
    sees only the prefix generated so far (so no future token can leak), and the
    argmax of the final position is appended. Decoding stops early at
    ``EOS_IDX`` or after ``max_len`` tokens. The returned tensor holds the
    predicted tokens (including ``EOS_IDX`` when it terminates) and has shape
    ``(pred_len,)``.
    '''
    if src.dim() != 2 or src.size(0) != 1:
        raise ValueError(
            'src must have shape (1, src_len); batch decoding is not supported'
        )
    if device is None:
        device = src.device
    model.eval()
    src = src.to(device)

    decoder_input = torch.tensor([[BOS_IDX]], dtype=torch.long, device=device)
    predicted: list[int] = []
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(src, decoder_input)
            next_token = int(logits[0, -1].argmax(dim=-1).item())
            predicted.append(next_token)
            if next_token == EOS_IDX:
                break
            decoder_input = torch.cat(
                (
                    decoder_input,
                    torch.tensor([[next_token]], dtype=torch.long, device=device),
                ),
                dim=1,
            )
    return torch.tensor(predicted, dtype=torch.long, device=device)


def evaluate(
    model: Transformer,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    '''Greedy-decode every example and report token and exact-match accuracy.

    Padding positions are ignored. Token accuracy counts individual correct
    tokens; exact-match accuracy counts examples whose full decoded sequence
    equals the unpadded labels.
    '''
    model.eval()
    total_tokens = 0
    total_correct = 0
    total_examples = 0
    total_exact = 0
    with torch.no_grad():
        for batch in loader:
            source = batch['source'].to(device)
            labels = batch['labels'].to(device)
            for i in range(source.size(0)):
                gold = labels[i]
                mask = gold != PAD_IDX
                n = int(mask.sum().item())
                if n == 0:
                    continue
                prediction = greedy_decode(
                    model, source[i : i + 1], max_len=n, device=device
                )
                gold = gold[:n]
                total_tokens += n
                total_examples += 1
                exact = prediction.size(0) == n
                for j in range(min(n, prediction.size(0))):
                    if prediction[j].item() == gold[j].item():
                        total_correct += 1
                    else:
                        exact = False
                if exact:
                    total_exact += 1
    token_accuracy = total_correct / total_tokens if total_tokens else 0.0
    exact_match_accuracy = total_exact / total_examples if total_examples else 0.0
    return {
        'token_accuracy': token_accuracy,
        'exact_match_accuracy': exact_match_accuracy,
    }


def train(config: TrainConfig) -> dict[str, Any]:
    '''Train the copy-task model and return final metrics and history.

    Runs a standard teacher-forcing loop with Adam and optional gradient
    clipping, evaluates every epoch, and saves the best checkpoint plus a JSON
    history and the flattened config into ``config.output_dir``.
    '''
    seed_everything(config.seed)
    device = resolve_device(config.device)
    model = build_model(config).to(device)
    train_loader, val_loader = build_dataloaders(config)

    use_noam = config.warmup_steps is not None and config.warmup_steps > 0
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0 if use_noam else config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )
    scheduler = _build_noam_scheduler(
        optimizer, config.d_model, config.warmup_steps
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_val_accuracy = -1.0
    best_epoch = -1
    start_time = time.perf_counter()

    epochs = config.epochs if config.epochs is not None else 1

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_tokens = 0
        for batch in train_loader:
            source = batch['source'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(source, decoder_input)
            loss = criterion(logits.permute(0, 2, 1), labels)
            loss.backward()
            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip_norm
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            mask = labels != PAD_IDX
            train_loss += loss.item() * mask.sum().item()
            train_tokens += mask.sum().item()

        train_loss = train_loss / train_tokens if train_tokens else 0.0
        val_loss, val_accuracy = _evaluate(model, val_loader, criterion, device)

        history.append(
            {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            save_checkpoint(
                output_dir / 'best.pt',
                model,
                optimizer,
                config,
                epoch,
                {'val_loss': val_loss, 'val_accuracy': val_accuracy},
            )

    elapsed = time.perf_counter() - start_time

    with open(output_dir / 'history.json', 'w', encoding='utf-8') as handle:
        json.dump(history, handle, indent=2)
    with open(output_dir / 'config.json', 'w', encoding='utf-8') as handle:
        json.dump(asdict(config), handle, indent=2)

    final = history[-1]
    return {
        'final_train_loss': final['train_loss'],
        'final_val_loss': final['val_loss'],
        'final_val_accuracy': final['val_accuracy'],
        'best_val_accuracy': best_val_accuracy,
        'best_epoch': best_epoch,
        'elapsed_seconds': elapsed,
        'history': history,
        'device': str(device),
    }
