import torch
import json
import os
import sys
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# CONFIGURATION
MODEL_PATH = "models/bert_sentiment"
TEST_DATA_PATH = "data/test.csv"  # Ensure you have this file!
METRICS_PATH = "metrics.json"
ACCURACY_THRESHOLD = 0.70  # The pipeline fails if accuracy is below 70%
BATCH_SIZE = 8


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model


def evaluate():
    print("🔄 Loading model and data...")
    tokenizer, model = load_model()

    # Load test data (Assuming CSV with 'text' and 'label' columns)
    # If your test file doesn't exist, we create dummy data to prevent crash during dev
    if os.path.exists(TEST_DATA_PATH):
        df = pd.read_csv(TEST_DATA_PATH)
    else:
        print(f"⚠️ Warning: {TEST_DATA_PATH} not found. Using dummy data for testing.")
        df = pd.DataFrame({
            "text": ["I love this", "I hate this", "This is great"],
            "label": ["positive", "negative", "positive"]
        })

    # Prepare inputs
    inputs = tokenizer(
        list(df["content"]),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # Predict
    print("🔮 Predicting...")
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    # Map text labels to integers if necessary (adjust based on your training)
    # Assuming model outputs 0=negative, 1=positive
    true_labels = [0 if score <= 2 else 1 for score in df["score"]]

    # Calculate Metrics
    acc = accuracy_score(true_labels, predictions.tolist())
    f1 = f1_score(true_labels, predictions.tolist(), average="weighted")

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save Metrics to JSON (Requirement: Store performance metrics)
    metrics = {"accuracy": acc, "f1_score": f1}
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f)

    # Fail pipeline if threshold not met (Requirement: Fail if below threshold)
    if acc < ACCURACY_THRESHOLD:
        print(f"❌ FAILED: Accuracy {acc:.2f} is below threshold {ACCURACY_THRESHOLD}")
        sys.exit(1)
    else:
        print(f"✅ PASSED: Accuracy {acc:.2f} meets threshold {ACCURACY_THRESHOLD}")


if __name__ == "__main__":
    evaluate()
