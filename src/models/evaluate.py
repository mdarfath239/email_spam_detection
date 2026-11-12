import json
import joblib
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.utils.config import (
    MODEL_PATH,
    TEST_DATA_PATH,
    MODEL_DIR,
    METRICS_PATH,
    CONFUSION_MATRIX_PATH,
)

def evaluate_model():
    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    X_test = test_df["clean_text"].astype(str)
    y_test = test_df["label"].astype(str)

    predictions = model.predict(X_test)

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=["ham", "spam"],
        zero_division=0,
    ))

    labels = ["ham", "spam"]
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    matrix_df = pd.DataFrame(
        matrix,
        index=["actual_ham", "actual_spam"],
        columns=["predicted_ham", "predicted_spam"],
    )

    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(matrix_df)

    summary = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "spam_precision": float(
            precision_score(y_test, predictions, pos_label="spam", zero_division=0)
        ),
        "spam_recall": float(
            recall_score(y_test, predictions, pos_label="spam", zero_division=0)
        ),
        "spam_f1": float(
            f1_score(y_test, predictions, pos_label="spam", zero_division=0)
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    matrix_df.to_csv(CONFUSION_MATRIX_PATH, index=True)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump({"final_test_metrics": summary}, file, indent=4)

    print("\nSaved evaluation artifacts in models/")
    print(summary)

if __name__ == "__main__":
    evaluate_model()