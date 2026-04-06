from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _make_dataset(inputs: np.ndarray, targets: np.ndarray) -> TensorDataset:
    return TensorDataset(torch.from_numpy(inputs).float(), torch.from_numpy(targets).float())


def create_dataloaders(
    data_path: str | Path,
    batch_size: int,
    seed: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_path = Path(data_path)
    with np.load(data_path) as data:
        train_dataset = _make_dataset(data["x_train"], data["y_train"])
        val_dataset = _make_dataset(data["x_val"], data["y_val"])
        test_dataset = _make_dataset(data["x_test"], data["y_test"])

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
