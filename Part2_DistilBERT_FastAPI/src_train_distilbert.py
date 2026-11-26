import time
import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import accuracy_score, f1_score


# -----------------------------
# Load data
# -----------------------------

train_df = pd.read_csv("data/train.csv")
validation_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")


# -----------------------------
# Label mapping
# -----------------------------

labels = sorted(train_df["category"].unique())

label2id = {
    label: index
    for index, label in enumerate(labels)
}

id2label = {
    index: label
    for label, index in label2id.items()
}


# -----------------------------
# Convert labels to numbers
# -----------------------------

train_df["label"] = train_df["category"].map(label2id)
validation_df["label"] = validation_df["category"].map(label2id)
test_df["label"] = test_df["category"].map(label2id)


# -----------------------------
# Create Hugging Face datasets
# -----------------------------

train_dataset = Dataset.from_pandas(
    train_df[["text", "label"]],
    preserve_index=False
)

validation_dataset = Dataset.from_pandas(
    validation_df[["text", "label"]],
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df[["text", "label"]],
    preserve_index=False
)


# -----------------------------
# Load tokenizer
# -----------------------------

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


def tokenize_data(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


train_dataset = train_dataset.map(
    tokenize_data,
    batched=True
)

validation_dataset = validation_dataset.map(
    tokenize_data,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_data,
    batched=True
)


# -----------------------------
# Load DistilBERT
# -----------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    label2id=label2id,
    id2label=id2label
)


# -----------------------------
# Metrics
# -----------------------------

def compute_metrics(eval_prediction):
    predictions, labels_true = eval_prediction

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        labels_true,
        predicted_labels
    )

    macro_f1 = f1_score(
        labels_true,
        predicted_labels,
        average="macro"
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }


# -----------------------------
# Training configuration
# -----------------------------

training_args = TrainingArguments(
    output_dir="results/distilbert_category",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    report_to="none"
)


# -----------------------------
# Trainer
# -----------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics
)


# -----------------------------
# Train
# -----------------------------

print("\nStarting DistilBERT category training...")

start_time = time.time()

trainer.train()

training_time = time.time() - start_time

print(
    f"\nTraining time: {training_time:.2f} seconds"
)


# -----------------------------
# Validation evaluation
# -----------------------------

validation_results = trainer.evaluate(
    validation_dataset
)

print("\nValidation Results:")
print(validation_results)


# -----------------------------
# Test evaluation
# -----------------------------

test_results = trainer.evaluate(
    test_dataset
)

print("\nTest Results:")
print(test_results)


# -----------------------------
# Inference timing
# -----------------------------

sample_texts = test_df["text"].tolist()

start_time = time.time()

predictions = trainer.predict(test_dataset)

inference_time = time.time() - start_time

inference_per_ticket = (
    inference_time / len(sample_texts)
)

print(
    f"\nTotal inference time: {inference_time:.4f} seconds"
)

print(
    f"Inference time per ticket: "
    f"{inference_per_ticket:.6f} seconds"
)


# -----------------------------
# Save model
# -----------------------------

model.save_pretrained(
    "results/distilbert_category_model"
)

tokenizer.save_pretrained(
    "results/distilbert_category_model"
)


# -----------------------------
# Final summary
# -----------------------------

print("\n==============================")
print("DISTILBERT CATEGORY SUMMARY")
print("==============================")

print(
    f"Validation Accuracy: "
    f"{validation_results['eval_accuracy']:.4f}"
)

print(
    f"Validation Macro F1: "
    f"{validation_results['eval_macro_f1']:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_results['eval_accuracy']:.4f}"
)

print(
    f"Test Macro F1: "
    f"{test_results['eval_macro_f1']:.4f}"
)

print(
    f"Training Time: "
    f"{training_time:.2f} seconds"
)

print(
    f"Inference Time/Ticket: "
    f"{inference_per_ticket:.6f} seconds"
)