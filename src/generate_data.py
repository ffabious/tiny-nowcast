from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


IMAGE_SIZE = 64
NUM_FRAMES = 5
MIN_BLOBS = 1
MAX_BLOBS = 3
VELOCITY_RANGE = (-2.0, 2.0)
SIGMA_RANGE = (3.0, 8.0)
AMPLITUDE_RANGE = (0.4, 1.0)
NOISE_STD = 0.03
DEFAULT_TRAIN_SIZE = 800
DEFAULT_EVAL_SIZE = 100
DEFAULT_OUTPUT = Path("data/synth_data.npz")


def gaussian_blob(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    center_x: float,
    center_y: float,
    sigma: float,
    amplitude: float,
) -> np.ndarray:
    squared_distance = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
    return amplitude * np.exp(-squared_distance / (2.0 * sigma**2))


def generate_sequence(rng: np.random.Generator) -> np.ndarray:
    grid_y, grid_x = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    frames = np.zeros((NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    num_blobs = rng.integers(MIN_BLOBS, MAX_BLOBS + 1)
    for _ in range(num_blobs):
        x0 = rng.uniform(0.0, IMAGE_SIZE - 1.0)
        y0 = rng.uniform(0.0, IMAGE_SIZE - 1.0)
        vx = rng.uniform(*VELOCITY_RANGE)
        vy = rng.uniform(*VELOCITY_RANGE)
        sigma = rng.uniform(*SIGMA_RANGE)
        amplitude = rng.uniform(*AMPLITUDE_RANGE)

        for t in range(NUM_FRAMES):
            frames[t] += gaussian_blob(
                grid_x=grid_x,
                grid_y=grid_y,
                center_x=x0 + vx * t,
                center_y=y0 + vy * t,
                sigma=sigma,
                amplitude=amplitude,
            )

    noise = rng.normal(0.0, NOISE_STD, size=frames.shape).astype(np.float32)
    return np.clip(frames + noise, 0.0, 1.0)


def generate_split(num_sequences: int, rng: np.random.Generator, name: str) -> tuple[np.ndarray, np.ndarray]:
    sequences = np.empty((num_sequences, NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    for i in tqdm(range(num_sequences), desc=f"Generating {name}", unit="seq"):
        sequences[i] = generate_sequence(rng)

    return sequences[:, :4], sequences[:, 4:5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic precipitation nowcasting data.")
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--val-size", type=int, default=DEFAULT_EVAL_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_EVAL_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    x_train, y_train = generate_split(args.train_size, rng, "train")
    x_val, y_val = generate_split(args.val_size, rng, "val")
    x_test, y_test = generate_split(args.test_size, rng, "test")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )

    print(f"Saved dataset to {args.output}")
    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"x_val: {x_val.shape}, y_val: {y_val.shape}")
    print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")


if __name__ == "__main__":
    main()
