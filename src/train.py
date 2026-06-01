"""
Train a CNN on MNIST and log everything to MLflow.

Usage:
    python src/train.py
    python src/train.py --epochs 10 --lr 0.001 --batch-size 128
"""

import argparse
import subprocess
import sys
import os

os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.model import MNISTClassifier

MODEL_NAME = "mnist-classifier"
EXPERIMENT_NAME = "mnist-digit-classifier"

NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def load_data(data_dir, batch_size, val_fraction=0.1):
    full_train = datasets.MNIST(data_dir, train=True, download=True, transform=NORMALIZE)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=NORMALIZE)

    n_val = int(len(full_train) * val_fraction)
    n_train = len(full_train) - n_val
    train_set, val_set = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    train_loader, val_loader, _ = load_data(args.data_dir, args.batch_size)

    model = MNISTClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    mlflow.enable_system_metrics_logging()
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        mlflow.log_params({
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "optimizer": "adam",
            "architecture": "2-layer-cnn",
        })
        mlflow.set_tags({
            "git_commit": get_git_commit(),
            "dataset": "MNIST",
            "dataset_version": "torchvision-v1",
            "framework": "pytorch",
        })

        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }, step=epoch)

            print(
                f"Epoch {epoch:02d}/{args.epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.4f}"
            )

        # Build a sample input for the model signature
        sample_input = next(iter(val_loader))[0][:4].to(device)
        sample_output = model(sample_input).detach().cpu().numpy()
        signature = mlflow.models.infer_signature(
            sample_input.cpu().numpy(),
            sample_output,
        )

        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=sample_input.cpu().numpy(),
            registered_model_name=MODEL_NAME,
        )

        print(f"\nRun ID: {run.info.run_id}")
        print(f"Model registered as '{MODEL_NAME}'")


def parse_args():
    p = argparse.ArgumentParser(description="Train MNIST classifier")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-dir", default="./data")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
