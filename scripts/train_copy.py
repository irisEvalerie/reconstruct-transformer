'''Command-line entry point for training the copy-task Transformer.'''

from __future__ import annotations

import argparse
from pathlib import Path

from reconstruct_transformer.training import TrainConfig, load_config, train


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
    args = parser.parse_args()

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
