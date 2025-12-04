import pytest
from unittest.mock import MagicMock, patch
import torch
import runpy

# Now that src.inference uses lazy loading, this import won't crash pytest
from src.inference import predict_sentiment


def test_predict_sentiment():
    # Example input
    sample_texts = ["This app is great!", "I hate this app."]

    # Mock tokenizer outputs (Must return tensors for the model to accept them)
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1], [2]]),
        "attention_mask": torch.tensor([[1], [1]]),
    }

    # Mock model output logits (Must be tensors for F.softmax)
    # [0.1, 0.9] -> index 1 (positive)
    # [0.8, 0.2] -> index 0 (negative)
    mock_model = MagicMock()
    mock_model.return_value.logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])

    # We patch 'load_trained_model' because 'get_model_and_tokenizer' calls it.
    with patch("src.inference.load_trained_model", return_value=(
        mock_model,
        mock_tokenizer
        )
    ):
        predictions = predict_sentiment(sample_texts)

    assert len(predictions) == 2
    assert predictions[0]["label"] == "positive"
    assert predictions[1]["label"] == "negative"
    assert "confidence" in predictions[0]


@patch("builtins.print")
@patch("src.data_processing.preprocess_text")
@patch("transformers.AutoModelForSequenceClassification.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_inference_main(mock_auto_tokenizer, mock_auto_model, mock_preprocess, mock_print):
    """
    Test the if __name__ == '__main__' block in src/inference.py
    """
    # 1. Setup Mocks
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.return_value = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.tensor([[1, 1], [1, 1]]),
    }
    
    mock_model_instance = MagicMock()
    # Logits need to be tensors
    mock_model_instance.return_value.logits = torch.tensor([
        [0.1, 0.9],  # -> Positive
        [0.8, 0.2]   # -> Negative
    ])
   
    # Configure the factories to return our instances
    mock_auto_tokenizer.return_value = mock_tokenizer_instance
    mock_auto_model.return_value = mock_model_instance
    
    # Mock preprocessing to return strings
    mock_preprocess.side_effect = lambda x: x 

    # 2. Run the script as __main__
    runpy.run_path("src/inference.py", run_name="__main__")

    # 3. Assertions
    # Note: runpy executes the whole file. Because of the lazy loading,
    # the main block calls 'predict_sentiment', which calls 'load_trained_model'.
    mock_auto_tokenizer.assert_called_once()
    mock_auto_model.assert_called_once()
    
    # Check if preprocess_text was called
    # (Based on the sample texts inside the __main__ block of inference.py)
    assert mock_preprocess.call_count >= 1
