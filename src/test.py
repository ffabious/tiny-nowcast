from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import matplotlib
import torch
from torch import nn

from dataset import create_dataloaders
from model import BaselineModel, TinyNowcastModel
from reproducibility import DEFAULT_SEED, set_seed

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    total_mse = 0.0
    total_mae = 0.0
    sample: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    model.eval()
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        predictions = model(inputs)

        total_mse += mse_loss(predictions, targets).item() * inputs.size(0)
        total_mae += mae_loss(predictions, targets).item() * inputs.size(0)

        if sample is None:
            sample = (
                inputs[0].detach().cpu(),
                targets[0].detach().cpu(),
                predictions[0].detach().cpu(),
            )

    if sample is None:
        raise ValueError("Evaluation loader is empty.")

    dataset_size = len(loader.dataset)
    return {
        "mse": total_mse / dataset_size,
        "mae": total_mae / dataset_size,
    }, sample


def load_models(model_path: Path, device: torch.device) -> OrderedDict[str, nn.Module]:
    models: OrderedDict[str, nn.Module] = OrderedDict()
    models["baseline"] = BaselineModel().to(device)

    if model_path.exists():
        tiny_model = TinyNowcastModel().to(device)
        state_dict = torch.load(model_path, map_location=device)
        tiny_model.load_state_dict(state_dict)
        models["tiny_nowcast"] = tiny_model

    return models


def save_loss_plot(metrics: OrderedDict[str, dict[str, dict[str, float]]], output_path: Path) -> None:
    model_names = list(metrics.keys())
    val_losses = [metrics[name]["val"]["mse"] for name in model_names]
    test_losses = [metrics[name]["test"]["mse"] for name in model_names]
    x_positions = list(range(len(model_names)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([x - width / 2 for x in x_positions], val_losses, width=width, label="val")
    ax.bar([x + width / 2 for x in x_positions], test_losses, width=width, label="test")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("MSE loss")
    ax.set_title("Validation and test loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_predictions_plot(
    inputs: torch.Tensor,
    target: torch.Tensor,
    predictions: OrderedDict[str, torch.Tensor],
    output_path: Path,
) -> None:
    panels: list[tuple[str, torch.Tensor]] = [("last_input", inputs[-1]), ("target", target[0])]
    panels.extend((name, prediction[0]) for name, prediction in predictions.items())

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image.numpy(), cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_metrics(metrics: OrderedDict[str, dict[str, dict[str, float]]], output_path: Path) -> None:
    lines: list[str] = []
    for model_name, splits in metrics.items():
        lines.append(model_name)
        for split_name, values in splits.items():
            lines.append(f"  {split_name}: mse={values['mse']:.6f}, mae={values['mae']:.6f}")
        lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate nowcasting models and save plots.")
    parser.add_argument("--data-path", type=str, default="data/synth_data.npz", help="Path to the dataset file.")
    parser.add_argument("--model-path", type=str, default="tiny_nowcast_model.pth", help="Path to a trained TinyNowcastModel state dict.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for evaluation artifacts.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducible evaluation.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, val_loader, test_loader = create_dataloaders(args.data_path, args.batch_size, seed=args.seed)
    models = load_models(Path(args.model_path), device)

    metrics: OrderedDict[str, dict[str, dict[str, float]]] = OrderedDict()
    prediction_panels: OrderedDict[str, torch.Tensor] = OrderedDict()
    sample_inputs: torch.Tensor | None = None
    sample_target: torch.Tensor | None = None

    for name, model in models.items():
        val_metrics, _ = evaluate_model(model, val_loader, device)
        test_metrics, sample = evaluate_model(model, test_loader, device)
        metrics[name] = {"val": val_metrics, "test": test_metrics}

        if sample_inputs is None or sample_target is None:
            sample_inputs, sample_target, _ = sample
        prediction_panels[name] = sample[2]

    if sample_inputs is None or sample_target is None:
        raise ValueError("No evaluation sample available.")

    save_loss_plot(metrics, output_dir / "loss.png")
    save_predictions_plot(sample_inputs, sample_target, prediction_panels, output_dir / "predictions.png")
    save_metrics(metrics, output_dir / "metrics.txt")
    print(f"Seed: {args.seed}")
    print(f"Saved evaluation outputs to {output_dir}")


if __name__ == "__main__":
    main()
