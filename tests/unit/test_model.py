import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from unittest.mock import patch, MagicMock
import numpy as np 
from src.model import train_model, compute_metrics, load_model_and_tokenizer
import runpy
import pandas as pd


def test_model_forward_pass():

    model_path = "models/bert_sentiment"  # path to the trained model

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Dummy input text (batch of 2)
    dummy_texts = [
        "I love this product! It's amazing.",
        "This app is terrible and does not work."
    ]

    # Tokenize to batch tensors
    inputs = tokenizer(
        dummy_texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # Run model forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits
    logits = outputs.logits

    # Assertions
    assert logits is not None, "Model did not return logits."
    assert len(logits.shape) == 2, "Logits should be a 2D tensor."
    assert logits.shape[0] == len(dummy_texts), "Batch size mismatch in logits."
    assert logits.shape[1] == model.config.num_labels, "Number of labels mismatch in logits."


@patch("src.model.Trainer")
@patch("src.model.load_model_and_tokenizer")
def test_train_model(mock_load_fn, mock_trainer_class):
    # Mock model + tokenizer correctly
    fake_model = MagicMock()
    fake_tokenizer = MagicMock()
    mock_load_fn.return_value = (fake_model, fake_tokenizer)

    # Mock Trainer
    mock_trainer = MagicMock()
    mock_trainer.train.return_value = None
    mock_trainer.save_model.return_value = None
    mock_trainer_class.return_value = mock_trainer

    # Dummy dataset
    dummy_data = [{"input_ids": [1, 2, 3], "labels": 1}]

    # Run the function being tested
    result = train_model(dummy_data, dummy_data, num_epochs=1)

    # Assertions
    assert result is not None
    mock_trainer.train.assert_called_once()
    mock_trainer.save_model.assert_called_once()
    fake_tokenizer.save_pretrained.assert_called_once()


def test_compute_metrics():
    """
    Ensure compute_metrics returns correct accuracy.
    """
    # Fake predictions where 3/4 match
    preds = MagicMock()
    preds.label_ids = np.array([0, 1, 1, 0])
    preds.predictions = np.array([
        [0.9, 0.1],   # -> class 0 (correct)
        [0.2, 0.8],   # -> class 1 (correct)
        [0.3, 0.7],   # -> class 1 (correct)
        [0.6, 0.4],   # -> class 0 (correct)
    ])

    result = compute_metrics(preds)
    assert "accuracy" in result
    assert result["accuracy"] == 1.0  # 4/4 correct
