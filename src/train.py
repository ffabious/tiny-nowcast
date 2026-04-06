import argparse
from pathlib import Path

import torch
from torch import nn, optim

from dataset import create_dataloaders
from model import TinyNowcastModel

def train_model(
    data_path: str,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    model_path: str,
    device: torch.device,
) -> None:
    train_loader, val_loader, _ = create_dataloaders(data_path, batch_size)

    model = TinyNowcastModel().to(device)
    criterion = nn.MSELoss()
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=learning_rate) if trainable_params else None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

    # Save the trained model
    model_path = Path(model_path)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TinyNowcastModel on synthetic precipitation data.")
    parser.add_argument("--data-path", type=str, default="data/synth_data.npz", help="Path to the dataset file.")
    parser.add_argument("--model-path", type=str, default="tiny_nowcast_model.pth", help="Path to save the trained model.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    train_model(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        model_path=args.model_path,
        device=device,
    )

if __name__ == "__main__":
    main()
