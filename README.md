# MNIST Digit Classifier with MLflow

A PyTorch CNN trained on MNIST with the full MLflow lifecycle wired in — experiment tracking, model registry, and a CI/CD quality gate via GitHub Actions.

## What This Project Demonstrates

- **Experiment tracking** — hyperparameters, per-epoch metrics, and system metrics (CPU/GPU) logged to MLflow
- **Model registry** — trained model versioned and registered automatically after each run
- **Promotion gate** — model is only aliased `champion` if test accuracy exceeds 98%
- **CI/CD** — GitHub Actions trains and evaluates on every push to `main`, failing the workflow if the model regresses

---

## Project Structure

```
├── src/
│   ├── model.py       # 2-layer CNN architecture
│   ├── train.py       # Training script with MLflow tracking
│   └── evaluate.py    # Evaluation and model promotion script
├── .github/
│   └── workflows/
│       └── train_and_promote.yml  # GitHub Actions CI/CD pipeline
├── Dockerfile
└── requirements.txt
```

---

## Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create and activate a virtual environment

```powershell
python -m venv mlflow
mlflow\Scripts\Activate.ps1
```

> If activation is blocked by execution policy, run first:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## Running Locally

### Train

```powershell
python src/train.py
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | `5` | Number of training epochs |
| `--batch-size` | `64` | Training batch size |
| `--lr` | `0.001` | Learning rate |
| `--data-dir` | `./data` | Directory to download MNIST into |

MNIST is downloaded automatically on the first run.

### Evaluate and promote

```powershell
python src/evaluate.py --threshold 0.98
```

Scores the latest registered model on the MNIST test set. If accuracy meets the threshold, the model version is tagged with the alias `champion` in the registry. Exits with code 1 otherwise.

### View results in the MLflow UI

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. Refresh the page after each training run to see new results.

---

## What Gets Logged

| Category | Metrics |
|----------|---------|
| Hyperparameters | `learning_rate`, `batch_size`, `epochs`, `optimizer`, `architecture` |
| Per-epoch | `train_loss`, `val_loss`, `val_accuracy` |
| Final | `test_accuracy` (logged by evaluate.py) |
| System | `cpu_utilization_percentage`, `system_memory_usage_megabytes`, `gpu_*` (if available) |
| Tags | `git_commit`, `dataset`, `framework` |

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/train_and_promote.yml`) runs on every push or pull request to `main`:

1. Installs Python 3.11 and dependencies
2. Trains the model for 5 epochs
3. Evaluates against the 98% accuracy threshold
4. Passes and tags the model as `champion` if the threshold is met
5. Fails the workflow if the model regresses, blocking the merge

The `mlflow.db` artifact is uploaded after each run so you can download and inspect it from the Actions tab.

---

## Docker

Build and run the full train + evaluate pipeline in a container:

```bash
docker build -t mnist-mlflow .
docker run --rm mnist-mlflow
```
