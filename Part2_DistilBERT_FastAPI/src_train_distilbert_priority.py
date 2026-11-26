import time
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import accuracy_score, f1_score


MODEL_NAME = "distilbert-base-uncased"


# Load datasets
train_df = pd.read_csv("data/train.csv")
validation_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")


# Create label mapping
labels = sorted(train_df["priority"].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

print("Priority labels:", labels)


# Convert labels to numbers
train_df["label"] = train_df["priority"].map(label2id)
validation_df["label"] = validation_df["priority"].map(label2id)
test_df["label"] = test_df["priority"].map(label2id)


# Keep required columns
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


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


train_dataset = train_dataset.map(tokenize, batched=True)
validation_dataset = validation_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)


# Load DistilBERT
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)


# Metrics
def compute_metrics(eval_pred):
    predictions, labels_true = eval_pred

    predictions = predictions.argmax(axis=-1)

    accuracy = accuracy_score(labels_true, predictions)

    macro_f1 = f1_score(
        labels_true,
        predictions,
        average="macro"
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }


# Training configuration
training_args = TrainingArguments(
    output_dir="results/distilbert_priority",
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


trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=validation_dataset,

    compute_metrics=compute_metrics
)


# Train
print()
print("==============================")
print("TRAINING DISTILBERT - PRIORITY")
print("==============================")

start_time = time.perf_counter()

trainer.train()

training_time = time.perf_counter() - start_time


# Validation evaluation
print()
print("Validation Results:")

validation_results = trainer.evaluate(
    eval_dataset=validation_dataset
)

print(validation_results)


# Test evaluation
print()
print("Test Results:")

test_results = trainer.evaluate(
    eval_dataset=test_dataset
)

print(test_results)


# Inference timing
start_inference = time.perf_counter()

predictions = trainer.predict(test_dataset)

inference_time = time.perf_counter() - start_inference

inference_per_ticket = (
    inference_time / len(test_dataset)
)


# Save model
trainer.save_model(
    "results/distilbert_priority_model"
)

tokenizer.save_pretrained(
    "results/distilbert_priority_model"
)


# Summary
print()
print("==============================")
print("DISTILBERT PRIORITY SUMMARY")
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