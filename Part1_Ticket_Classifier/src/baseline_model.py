import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# Load datasets
train_df = pd.read_csv("data/train.csv")
validation_df = pd.read_csv("data/validation.csv")

X_train = train_df["text"]
X_validation = validation_df["text"]


def train_model(target):
    y_train = train_df[target]
    y_validation = validation_df[target]

    # TF-IDF + Logistic Regression pipeline
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english"
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

    # Tune ngram_range and C
    parameters = {
        "tfidf__ngram_range": [
            (1, 1),
            (1, 2)
        ],
        "model__C": [
            0.5,
            1,
            2
        ]
    }

    grid = GridSearchCV(
        pipeline,
        parameters,
        cv=3,
        scoring="f1_macro",
        n_jobs=-1
    )

    # Train model
    grid.fit(X_train, y_train)

    # Validation predictions
    predictions = grid.predict(X_validation)

    # Metrics
    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro"
    )

    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=grid.classes_
    )

    print("\n==============================")
    print(f"Target: {target}")
    print("==============================")

    print("Best parameters:", grid.best_params_)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClasses:")
    print(grid.classes_)

    # Save model
    model_file = f"src/models/{target.lower()}_baseline.joblib"

    joblib.dump(
        grid.best_estimator_,
        model_file
    )

    print(f"\nModel saved to: {model_file}")


# Train category model
train_model("category")

# Train priority model
train_model("priority")