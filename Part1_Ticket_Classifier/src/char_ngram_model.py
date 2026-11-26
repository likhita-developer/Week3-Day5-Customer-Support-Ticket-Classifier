import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# Load datasets
train_df = pd.read_csv("data/train.csv")
validation_df = pd.read_csv("data/validation.csv")


def train_character_model(target):
    X_train = train_df["text"]
    y_train = train_df[target]

    X_validation = validation_df["text"]
    y_validation = validation_df[target]

    # Character n-gram TF-IDF + Logistic Regression
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5)
            )
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Make predictions
    predictions = pipeline.predict(X_validation)

    # Calculate metrics
    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro"
    )

    print("\n==============================")
    print(f"Character n-gram model: {target}")
    print("==============================")

    print("Analyzer: char")
    print("N-gram range: (3, 5)")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Macro F1: {macro_f1:.4f}")


# Train category model
train_character_model("category")

# Train priority model
train_character_model("priority")