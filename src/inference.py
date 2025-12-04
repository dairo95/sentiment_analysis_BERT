import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.data_processing import preprocess_text

MODEL_DIR = "models/bert_sentiment"

# Auto device detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables for caching (Lazy Loading)
_model = None
_tokenizer = None


def load_trained_model(model_dir: str = MODEL_DIR):
    """
    Loads the model and tokenizer from disk.
    This function performs the heavy lifting and I/O.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    model.to(device)
    return model, tokenizer


def get_model_and_tokenizer():
    """
    Singleton accessor.
    Only loads the model the first time it is called.
    Prevents global scope execution crashes.
    """
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        _model, _tokenizer = load_trained_model(MODEL_DIR)
    return _model, _tokenizer


def predict_sentiment(texts):
    if isinstance(texts, str):
        texts = [texts]

    # Clean
    texts = [preprocess_text(t) for t in texts]

    # --- LAZY LOAD START ---
    # We call the getter here instead of using global variables directly.
    model, tokenizer = get_model_and_tokenizer()
    # --- LAZY LOAD END ---

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
