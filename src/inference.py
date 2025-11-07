import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.data_processing import preprocess_text  # reuse training preprocessing

MODEL_DIR = "models/bert_sentiment"

# Auto device detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(model_dir: str = MODEL_DIR):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    model.to(device)
    return model, tokenizer


# Load once globally
model, tokenizer = load_trained_model(MODEL_DIR)


def predict_sentiment(texts):
    if isinstance(texts, str):
        texts = [texts]

    # Clean
    texts = [preprocess_text(t) for t in texts]

    # Tokenize
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

    LABELS = ["negative", "positive"]
    results = []
    for text, pred, prob in zip(texts, preds, probs):
        label = LABELS[pred.item()]
        confidence = prob[pred].item()
        results.append({
            "text": text,
            "label": label,
            "confidence": round(confidence, 4)
        })
    return results


if __name__ == "__main__":
    sample_texts = [
        "This is not a good product.",
        "I had a terrible experience with this item."
    ]
    predictions = predict_sentiment(sample_texts)
    for pred in predictions:
        print(f"📝 Text: {pred['text']}")
        print(f"➡  Predicted Sentiment: {pred['label']} (Confidence: {pred['confidence']})")
        print("-" * 60)