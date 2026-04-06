from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

from reproducibility import DEFAULT_SEED


def run_step(name: str, command: list[str], cwd: Path, seed: int) -> None:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    print(f"[pipeline] Running {name}: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=cwd, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate data, train the model, and run evaluation.")
    parser.add_argument("--data-path", type=str, default="data/synth_data.npz", help="Path to save the generated dataset.")
    parser.add_argument("--model-path", type=str, default="tiny_nowcast_model.pth", help="Path to save the trained model.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for evaluation artifacts.")
    parser.add_argument("--train-size", type=int, default=800, help="Number of training sequences to generate.")
    parser.add_argument("--val-size", type=int, default=100, help="Number of validation sequences to generate.")
    parser.add_argument("--test-size", type=int, default=100, help="Number of test sequences to generate.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training and evaluation.")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for training.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Shared random seed for all pipeline stages.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training and evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    python_executable = sys.executable
    src_dir = repo_root / "src"

    run_step(
        "data generation",
        [
            python_executable,
            str(src_dir / "generate_data.py"),
            "--output",
            args.data_path,
            "--train-size",
            str(args.train_size),
            "--val-size",
            str(args.val_size),
            "--test-size",
            str(args.test_size),
            "--seed",
            str(args.seed),
        ],
        repo_root,
        args.seed,
    )
    run_step(
        "training",
        [
            python_executable,
            str(src_dir / "train.py"),
            "--data-path",
            args.data_path,
            "--model-path",
            args.model_path,
            "--batch-size",
            str(args.batch_size),
            "--num-epochs",
            str(args.num_epochs),
            "--learning-rate",
            str(args.learning_rate),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ],
        repo_root,
        args.seed,
    )
    run_step(
        "evaluation",
        [
            python_executable,
            str(src_dir / "test.py"),
            "--data-path",
            args.data_path,
            "--model-path",
            args.model_path,
            "--output-dir",
            args.output_dir,
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ],
        repo_root,
        args.seed,
    )
    print("[pipeline] Done.", flush=True)


if __name__ == "__main__":
    main()
