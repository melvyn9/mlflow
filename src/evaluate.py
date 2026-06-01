"""
Evaluate the latest registered model version against the test set.
Promotes it to alias 'champion' if accuracy >= threshold, else exits 1
(which fails the GitHub Actions workflow).

Usage:
    python src/evaluate.py
    python src/evaluate.py --threshold 0.98
"""

import argparse
import sys
import os

import mlflow
import mlflow.pytorch
import torch
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

MODEL_NAME = "mnist-classifier"
DEFAULT_THRESHOLD = 0.98

NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def get_latest_version(client, model_name):
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        print(f"ERROR: No registered versions found for model '{model_name}'")
        sys.exit(1)
    return max(versions, key=lambda v: int(v.version))


@torch.no_grad()
def score_on_test_set(model, data_dir, device):
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=NORMALIZE)
    loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

    model.eval()
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
    return correct / len(test_set)


def evaluate(args):
    client = MlflowClient()
    latest = get_latest_version(client, MODEL_NAME)
    model_uri = f"models:/{MODEL_NAME}/{latest.version}"

    print(f"Loading model '{MODEL_NAME}' version {latest.version} (run: {latest.run_id})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mlflow.pytorch.load_model(model_uri, map_location=device)

    accuracy = score_on_test_set(model, args.data_dir, device)
    print(f"Test accuracy : {accuracy:.4f}")
    print(f"Threshold     : {args.threshold:.4f}")

    # Write result back to the originating run so it's visible in the UI
    with mlflow.start_run(run_id=latest.run_id):
        mlflow.log_metric("test_accuracy", accuracy)

    if accuracy >= args.threshold:
        client.set_registered_model_alias(MODEL_NAME, "champion", latest.version)
        print(f"PASS — model v{latest.version} promoted to alias 'champion'")
    else:
        print(f"FAIL — accuracy {accuracy:.4f} is below threshold {args.threshold:.4f}")
        sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate and promote MNIST model")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--data-dir", default="./data")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
