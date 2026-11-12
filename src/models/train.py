import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.features.vectorize import create_vectorizer
from src.utils.config import (
    PROCESSED_DATA_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    METRICS_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)

def train_models():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    X = df["clean_text"].astype(str)
    y = df["label"].astype(str)

    if y.nunique() != 2:
        raise ValueError("Expected both ham and spam labels.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Save the exact split so evaluation can reproduce the held-out test.
    TRAIN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"clean_text": X_train, "label": y_train}).to_csv(
        TRAIN_DATA_PATH, index=False
    )
    pd.DataFrame({"clean_text": X_test, "label": y_test}).to_csv(
        TEST_DATA_PATH, index=False
    )

    candidates = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
    }

    results = []
    best_score = -1.0
    best_pipeline = None

    for name, classifier in candidates.items():
        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ("tfidf", create_vectorizer()),
            ("classifier", classifier),
        ])

        # TF-IDF is fitted only on X_train inside this pipeline.
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        metrics = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "spam_precision": float(
                precision_score(
                    y_test, predictions,
                    pos_label="spam", zero_division=0
                )
            ),
            "spam_recall": float(
                recall_score(
                    y_test, predictions,
                    pos_label="spam", zero_division=0
                )
            ),
            "spam_f1": float(
                f1_score(
                    y_test, predictions,
                    pos_label="spam", zero_division=0
                )
            ),
        }

        print(metrics)
        results.append(metrics)

        if metrics["spam_f1"] > best_score:
            best_score = metrics["spam_f1"]
            best_pipeline = pipeline

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    pd.DataFrame(results).to_csv(
        MODEL_DIR / "model_comparison.csv",
        index=False,
    )

    print("\nTraining complete.")
    print("Best model pipeline saved to:", MODEL_PATH)
    print(pd.DataFrame(results))

if __name__ == "__main__":
    train_models()