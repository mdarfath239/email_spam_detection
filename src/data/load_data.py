import pandas as pd
from src.utils.config import RAW_DATA_PATH, TEXT_COLUMN, LABEL_COLUMN

def load_data():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}\n"
            "Put the Kaggle CSV inside data/raw/."
        )

    df = pd.read_csv(RAW_DATA_PATH)

    print("\nFirst five rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nData types:")
    print(df.dtypes)

    print("\nUnique label values (check the label column name first):")
    if LABEL_COLUMN in df.columns:
        print(df[LABEL_COLUMN].value_counts(dropna=False))

    return df

if __name__ == "__main__":
    load_data()