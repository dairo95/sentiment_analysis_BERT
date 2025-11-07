from unittest.mock import MagicMock, patch
from src.inference import predict_sentiment
import runpy
import torch


def test_predict_sentiment():
    # Example input
    sample_texts = ["This app is great!", "I hate this app."]

    # Mock tokenizer outputs
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": [["id1"], ["id2"]],
        "attention_mask": [[1], [1]],
    }

    # Mock model output logits
    mock_model = MagicMock()
    mock_model.return_value.logits = [[0.1, 0.9], [0.8, 0.2]]

    with patch("src.inference.load_trained_model", return_value=(mock_model, mock_tokenizer)):
        predictions = predict_sentiment(sample_texts)

    assert len(predictions) == 2
    assert predictions[0]["label"] in ["positive", "negative"]
    assert predictions[1]["label"] in ["positive", "negative"]


@patch("builtins.print")
@patch("src.data_processing.preprocess_text")
@patch("transformers.AutoModelForSequenceClassification.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_inference_main(mock_auto_tokenizer, mock_auto_model, mock_preprocess, mock_print):
    """
    Test the if __name__ == '__main__' block in src/inference.py
    by patching its external dependencies.
    """
    # 1. Setup Mocks
    # Mock what load_trained_model depends on
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.return_value = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.tensor([[1, 1], [1, 1]]),
    }
    mock_model_instance = MagicMock()
    mock_model_instance.return_value.logits = torch.tensor([
        [0.1, 0.9],  # -> 1 (positive)
        [0.8, 0.2]   # -> 0 (negative)
    ])
    mock_auto_tokenizer.return_value = mock_tokenizer_instance
    mock_auto_model.return_value = mock_model_instance
    
    # Mock what predict_sentiment depends on
    mock_preprocess.side_effect = lambda x: x  # Pass through text

    # 2. Run the script as __main__
    runpy.run_path("src/inference.py", run_name="__main__")

    # 3. Assertions
    # Check if model/tokenizer were loaded
    mock_auto_tokenizer.assert_called_once()
    mock_auto_model.assert_called_once()
    
    # Check if preprocess_text was called
    mock_preprocess.assert_any_call("This is not a good product.")
    mock_preprocess.assert_any_call("I had a terrible experience with this item.")
