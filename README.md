# Email Spam Detection

A machine learning project for classifying emails as `spam` or `ham` using a text-processing pipeline and a trained classifier. The project includes data preprocessing, model training, evaluation, and a FastAPI endpoint for inference.

## Overview

This project trains a spam detection model on email text using:

- text normalization and cleaning
- token-level preprocessing (stop-word removal and stemming)
- TF-IDF feature extraction
- model comparison between `Multinomial Naive Bayes` and `Logistic Regression`
- saved model artifacts and evaluation metrics

## Project Structure

```text
email_spam_detection/
├── data/
│   ├── raw/
│   │   └── spam_ham_dataset.csv
│   └── processed/
│       ├── cleaned_emails.csv
│       ├── train_emails.csv
│       └── test_emails.csv
├── models/
│   ├── confusion_matrix.csv
│   ├── evaluation_metrics.json
│   ├── model_comparison.csv
│   └── spam_ham_pipeline.joblib
├── notebooks/
├── src/
│   ├── api/
│   │   └── app.py
│   ├── data/
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   └── process_dataset.py
│   ├── features/
│   │   └── vectorize.py
│   ├── models/
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── train.py
│   └── utils/
│       └── config.py
├── requirements.txt
├── README.md
└── .venv/
```

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- NLTK
- joblib
- FastAPI
- Uvicorn

## Data Pipeline

The workflow is designed as follows:

1. Load the raw dataset from `data/raw/spam_ham_dataset.csv`
2. Validate and normalize labels (`ham`, `spam`, `0`, `1`)
3. Clean and preprocess email text using `src/data/preprocess.py`
4. Save processed data to `data/processed/cleaned_emails.csv`
5. Split into training and testing sets
6. Train candidate models using TF-IDF features
7. Select the best-performing pipeline based on spam F1-score
8. Save the trained model and evaluation outputs under `models/`

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Pipeline

### 1) Preprocess the dataset

```bash
python -m src.data.process_dataset
```

This creates the cleaned dataset in `data/processed/cleaned_emails.csv`.

### 2) Train the model

```bash
python -m src.models.train
```

This trains the classifiers, compares them, and saves the best model to:

```text
models/spam_ham_pipeline.joblib
```

### 3) Evaluate the model

```bash
python -m src.models.evaluate
```

This writes evaluation metrics and confusion matrix files under `models/`.

## API Usage

The project includes a FastAPI service for prediction.

Start the API:

```bash
uvicorn src.api.app:app --reload
```

The app exposes:

- `GET /` — health check
- `POST /predict` — predict whether an email is spam or ham

### Example request

```json
{
  "email_text": "Congratulations! You have won a free prize. Click here to claim your reward now."
}
```

### Example response

```json
{
  "prediction": "SPAM"
}
```

## Output Artifacts

The trained model and metrics are saved in the `models/` directory:

- `spam_ham_pipeline.joblib` — serialized trained pipeline
- `evaluation_metrics.json` — summary metrics
- `model_comparison.csv` — comparison of candidate models
- `confusion_matrix.csv` — confusion matrix for evaluation

## Notes

- The dataset is cleaned using custom text normalization for URLs, email addresses, numbers, punctuation, stop words, and stemming.
- The current model pipeline uses TF-IDF with bigrams and a balanced logistic regression or multinomial naive Bayes classifier depending on which performed best during training.
- The project is modular, so the data processing, feature engineering, model training, and API layers can be extended independently.

## License

This project is intended for educational and demonstration purposes. Update this section if you plan to distribute it under a specific license.
