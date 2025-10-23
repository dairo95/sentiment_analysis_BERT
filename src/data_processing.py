import re
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset

def clean_text(text: str) -> str:
    """
    Cleans the input text by lowercasing and removing special characters.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def preprocess_and_tokenize(df: pd.DataFrame, model_name: str = "bert-base-uncased"):
    """
    Cleans, splits, and tokenizes the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame with 'text' and 'label' columns.
        model_name (str): The name of the Hugging Face model to use for tokenization.

    Returns:
        tuple: A tuple containing tokenized train and validation datasets.
    """
    # 1. Clean text
    df['text'] = df['text'].apply(clean_text)

    # 2. Split data into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # 3. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # Convert pandas DataFrame to Hugging Face Dataset object
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # 4. Tokenize the datasets
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Set format for PyTorch
    tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    tokenized_val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    print("Data cleaning, splitting, and tokenization complete.")
    return tokenized_train_dataset, tokenized_val_dataset

