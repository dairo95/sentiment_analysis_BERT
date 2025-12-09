import torch
import json
import os
import sys
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# CONFIGURATION
# Local path on your laptop
MODEL_PATH = "models/bert_sentiment"
# Fallback model from Hugging Face (similar to yours) for GitHub Actions
FALLBACK_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

TEST_DATA_PATH = "data/test.csv"
METRICS_PATH = "metrics.json"
ACCURACY_THRESHOLD = 0.60
BATCH_SIZE = 16


def load_model():
    """
    Tries to load the model locally. If not found (e.g., on GitHub Actions),
    it downloads a fallback model from Hugging Face.
    """
    if os.path.exists(MODEL_PATH):
        print(f"📂 Found local model at {MODEL_PATH}. Loading...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    else:
        print(f"⚠️ Local model not found at {MODEL_PATH} (Expected on GitHub Actions).")
        print(f"🌐 Downloading fallback model: {FALLBACK_MODEL}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(FALLBACK_MODEL)
        except Exception as e:
            print(f"❌ Failed to download fallback model: {e}")
            sys.exit(1)

    return tokenizer, model


def evaluate():
    print("🔄 Starting Evaluation...")
    tokenizer, model = load_model()
    model.eval()

    # 1. Load Data
    if os.path.exists(TEST_DATA_PATH):
        df = pd.read_csv(TEST_DATA_PATH)
    else:
        print(f"⚠️ Warning: {TEST_DATA_PATH} not found. Using dummy data.")
        df = pd.DataFrame({
            "content": ["I love this project", "This is terrible", "Not bad"],
            "score": [5, 1, 3]
        })

    print(f"📊 Loaded {len(df)} test samples.")

    # 2. Prediction Loop (Batched)
    texts = list(df["content"])
    all_predictions = []

    print(f"🔮 Predicting in batches of {BATCH_SIZE}...")

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i: i + BATCH_SIZE]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(batch_preds.tolist())

        if (i // BATCH_SIZE) % 10 == 0:
            print(f"   Processed {min(i + BATCH_SIZE, len(texts))}/{len(texts)}...")

    # 3. Map Scores to Binary Labels
    # Fallback model and your model might map differently, but standard is 0=Neg, 1=Pos
    true_labels = [0 if score <= 2 else 1 for score in df["score"]]

    # 4. Metrics
    acc = accuracy_score(true_labels, all_predictions)
    f1 = f1_score(true_labels, all_predictions, average="weighted")

    print(f"📊 Accuracy: {acc:.4f}")
    print(f"📉 F1 Score: {f1:.4f}")

    # 5. Save Metrics
    metrics = {"accuracy": acc, "f1_score": f1}
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f)

    # 6. Quality Gate
    if acc < ACCURACY_THRESHOLD:
        print(f"❌ FAILED: Accuracy {acc:.2f} is below threshold {ACCURACY_THRESHOLD}")
        sys.exit(1)
    else:
        print(f"✅ PASSED: Accuracy {acc:.2f} meets threshold {ACCURACY_THRESHOLD}")
        sys.exit(0)


if __name__ == "__main__":
    evaluate()
