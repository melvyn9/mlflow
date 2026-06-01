FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Default: train then evaluate
CMD ["sh", "-c", "python src/train.py && python src/evaluate.py"]
