import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset  # If using custom datasets


def load_model_and_tokenizer(model_name: str = "bert-base-uncased",
                             num_labels: int = 2):
    """
    Loads a pretrained BERT model and tokenizer for sequence classification.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               num_labels=num_labels)
    return model, tokenizer


def compute_metrics(pred):
    """
    Compute accuracy for evaluation.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}


def train_model(
    train_dataset,
    val_dataset,
    model_name: str = "distilbert-base-uncased", # lighter model for faster training
    num_epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
):
    """
    Fine-tunes a pretrained BERT model on the given
    tokenized dataset using Trainer.
    """
    model, tokenizer = load_model_and_tokenizer(model_name)

    # Ensure the save directory exists
    os.makedirs("models/bert_sentiment", exist_ok=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="models/bert_sentiment",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),  # use mixed precision if GPU is
                                         # available
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model("models/bert_sentiment")
    tokenizer.save_pretrained("models/bert_sentiment")
    print("💾 Model saved to models/bert_sentiment")

    return model


if __name__ == "__main__":
    from data_extraction import load_data
    from data_processing import preprocess_and_tokenize

    # 1. Load your dataset
    df = load_data("data/dataset.csv")  # 🔁 adjust if file is elsewhere

    # 2. Keep only the relevant columns
    df = df[["content", "score"]].rename(columns={"content": "text", "score": "label"})

    # 3. Convert scores (1–5) to binary labels
    #    1–2 → negative (0), 4–5 → positive (1), drop 3 (neutral)
    df = df[df["label"] != 3]  # remove neutral examples
    df["label"] = df["label"].apply(lambda x: 1 if x > 3 else 0)

    print(df.head())  # optional sanity check

    # 4. Preprocess and tokenize
    train_dataset, val_dataset = preprocess_and_tokenize(df)

    # 5. Train and save
    train_model(train_dataset, val_dataset)
