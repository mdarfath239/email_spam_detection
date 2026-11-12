import pandas as pd

from src.data.preprocess import clean_batch
from src.utils.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    TEXT_COLUMN,
    LABEL_COLUMN,
)

LABEL_MAPPING = {
    "ham": "ham",
    "spam": "spam",
    "0": "ham",
    "1": "spam",
}

def process_dataset():
    print("Process started")

    df = pd.read_csv(RAW_DATA_PATH)
    print("Dataset loaded:", df.shape)

    required = {TEXT_COLUMN, LABEL_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Available: {df.columns.tolist()}"
        )

    df = df[[TEXT_COLUMN, LABEL_COLUMN]].copy()
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])

    df["email_text"] = df[TEXT_COLUMN].astype(str).str.strip()
    raw_labels = df[LABEL_COLUMN].astype(str).str.lower().str.strip()
    df["label"] = raw_labels.map(LABEL_MAPPING)

    unknown_labels = sorted(raw_labels[df["label"].isna()].unique().tolist())
    if unknown_labels:
        raise ValueError(
            f"Unknown labels: {unknown_labels}. Update LABEL_MAPPING."
        )

    df = df[df["email_text"].str.len() > 0]
    df = df.drop_duplicates(subset=["email_text"]).reset_index(drop=True)

    print("Rows after validation and duplicate removal:", len(df))
    print("Class distribution:")
    print(df["label"].value_counts())

    print("Cleaning email text...")
    df["clean_text"] = clean_batch(df["email_text"].tolist())

    # Remove rows that became empty after cleaning.
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[["email_text", "clean_text", "label"]].to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )

    print("Processed dataset saved:", PROCESSED_DATA_PATH)
    print("Final shape:", df.shape)
    print(df[["email_text", "clean_text", "label"]].head())

if __name__ == "__main__":
    process_dataset()