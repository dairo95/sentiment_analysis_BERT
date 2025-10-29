# tests/unit/test_data_extraction.py

import pytest
import pandas as pd
from pathlib import Path

# import the function from your code
from src.data_extraction import load_data


def test_load_data_valid_file(tmp_path):
    """
    ✅ Test 1: Verify a valid CSV file loads correctly.
    """
    # 1️⃣ Create a small fake CSV file
    data = pd.DataFrame({
        "text": ["I love AI", "Bad day", "Great movie"],
        "label": [1, 0, 1]
    })
    test_file = tmp_path / "sample.csv"
    data.to_csv(test_file, index=False)

    # 2️⃣ Use your load_data() function
    df = load_data(test_file)

    # 3️⃣ Check that it works correctly
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["text", "label"]
    assert len(df) == 3


def test_load_data_missing_file():
    """
    ✅ Test 2: Check if missing file raises FileNotFoundError.
    """
    fake_path = "data/missing.csv"

    with pytest.raises(FileNotFoundError):
        load_data(fake_path)


def test_load_data_empty_file(tmp_path):
    """
    ✅ Test 3: Handle an empty CSV file correctly.
    """
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")  # make an empty file

    with pytest.raises(pd.errors.EmptyDataError):
        load_data(empty_file)
