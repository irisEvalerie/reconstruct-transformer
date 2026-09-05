'''Command-line entry point for training and evaluating the copy-task model.'''

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from reconstruct_transformer.data import PAD_IDX
from reconstruct_transformer.training import (
    TrainConfig,
    build_dataloaders,
    evaluate,
    greedy_decode,
    load_config,
    load_model_from_checkpoint,
    resolve_device,
    train,
)


def quick_config() -> TrainConfig:
    '''A tiny configuration for fast smoke runs and CI.'''
    return TrainConfig(
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
        epochs=150,
        device='auto',
        output_dir='outputs/copy_task_quick',
    )


def _demo(model, loader, device, count: int = 4) -> None:
    '''Print input / target / predicted for a few validation examples.'''
    model.eval()
    shown = 0
    with torch.no_grad():
        for batch in loader:
            source = batch['source'].to(device)
            labels = batch['labels'].to(device)
            for i in range(source.size(0)):
                if shown >= count:
                    return
                source_len = int((source[i] != PAD_IDX).sum().item())
                label_len = int((labels[i] != PAD_IDX).sum().item())
                prediction = greedy_decode(
                    model, source[i : i + 1], max_len=label_len, device=device
                )
                print(f"input   {source[i, :source_len].tolist()}")
                print(f"target  {labels[i, :label_len].tolist()}")
                print(f"predict {prediction.tolist()}")
                print()
                shown += 1


def _run_eval(checkpoint: str, device: str | None, demo: bool) -> None:
    model, config, epoch, _ = load_model_from_checkpoint(checkpoint)
    device = resolve_device(device or config.device)
    model = model.to(device)
    _, val_loader = build_dataloaders(config)

    result = evaluate(model, val_loader, device)
    print(f"checkpoint:        {checkpoint}")
    print(f"device:            {device}")
    print(f"trained epochs:    {epoch}")
    print(f"token accuracy:    {result['token_accuracy']:.4f}")
    print(f"exact-match acc:   {result['exact_match_accuracy']:.4f}")

    eval_path = Path(config.output_dir) / 'eval.json'
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, 'w', encoding='utf-8') as handle:
        json.dump({'epoch': epoch, **result}, handle, indent=2)
    print(f"eval result:       {eval_path}")

    if demo:
        print("\n-- example predictions --")
        _demo(model, val_loader, device)


def main() -> None:
    parser = argparse.ArgumentParser(description='Train the copy-task model.')
    parser.add_argument(
        '--config', default='configs/copy_task.yaml', help='path to a YAML config'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='run a tiny config instead of loading a YAML file',
    )
    parser.add_argument(
        '--device', default=None, help="override device ('auto', 'cpu', 'cuda')"
    )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='evaluate a saved checkpoint instead of training',
    )
    parser.add_argument(
        '--checkpoint', default=None, help='checkpoint path for --eval-only'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='print a few example predictions during evaluation',
    )
    args = parser.parse_args()

    if args.eval_only:
        if not args.checkpoint:
            parser.error('--eval-only requires --checkpoint')
        _run_eval(args.checkpoint, args.device, args.demo)
        return

    config = quick_config() if args.quick else load_config(args.config)
    if args.device is not None:
        config.device = args.device

    result = train(config)

    print(f"device:            {result['device']}")
    print(f"elapsed:           {result['elapsed_seconds']:.2f}s")
    print(f"final train loss:  {result['final_train_loss']:.4f}")
    print(f"final val loss:    {result['final_val_loss']:.4f}")
    print(f"final val acc:     {result['final_val_accuracy']:.4f}")
    print(f"best val acc:      {result['best_val_accuracy']:.4f}")
    print(f"best epoch:        {result['best_epoch']}")
    print(f"checkpoint:        {Path(config.output_dir) / 'best.pt'}")
    print(f"history:           {Path(config.output_dir) / 'history.json'}")


if __name__ == '__main__':
    main()
