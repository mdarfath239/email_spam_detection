from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "spam_ham_dataset.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_emails.csv"
TRAIN_DATA_PATH = BASE_DIR / "data" / "processed" / "train_emails.csv"
TEST_DATA_PATH = BASE_DIR / "data" / "processed" / "test_emails.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "spam_ham_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"
CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix.csv"

# Update these two values after inspecting the CSV.
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

RANDOM_STATE = 42
TEST_SIZE = 0.20