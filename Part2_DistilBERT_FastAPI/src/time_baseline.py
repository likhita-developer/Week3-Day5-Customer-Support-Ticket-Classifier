import time
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


# Load data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

X_train = train["text"]
y_train = train["category"]

X_test = test["text"]
y_test = test["category"]


# Create baseline model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )),
    ("model", LogisticRegression(
        max_iter=1000
    ))
])


# Measure training time
start_train = time.perf_counter()

model.fit(X_train, y_train)

training_time = time.perf_counter() - start_train


# Measure inference time
start_inference = time.perf_counter()

predictions = model.predict(X_test)

inference_time = time.perf_counter() - start_inference
inference_per_ticket = inference_time / len(X_test)


# Metrics
accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(y_test, predictions, average="macro")


print()
print("==============================")
print("BASELINE CATEGORY RESULTS")
print("==============================")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Macro F1: {macro_f1:.4f}")
print(f"Training Time: {training_time:.4f} seconds")
print(f"Total Inference Time: {inference_time:.4f} seconds")
print(f"Inference Time per Ticket: {inference_per_ticket:.6f} seconds")