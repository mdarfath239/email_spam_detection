import joblib
from src.data.preprocess import clean_text
from src.utils.config import MODEL_PATH

def predict_email(email_text):
    model = joblib.load(MODEL_PATH)

    cleaned = clean_text(email_text)
    prediction = model.predict([cleaned])[0]

    return str(prediction).upper()

if __name__ == "__main__":
    print("Email Spam / Ham Predictor")
    print("Type 'exit' to stop.")

    while True:
        email_text = input("\nEnter email text: ").strip()

        if email_text.lower() == "exit":
            break

        if not email_text:
            print("Please enter some email text.")
            continue

        result = predict_email(email_text)
        print("Prediction:", result)