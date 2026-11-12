import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.preprocess import clean_text
from src.utils.config import MODEL_PATH

app = FastAPI(
    title="Email Spam / Ham Detection API",
    description="NLP text classifier",
    version="1.0.0",
)

if not MODEL_PATH.exists():
    raise RuntimeError(
        "Trained model not found. Run: python -m src.models.train"
    )

model = joblib.load(MODEL_PATH)

class EmailRequest(BaseModel):
    email_text: str = Field(min_length=1, max_length=50000)

@app.get("/")
def home():
    return {"message": "Email Spam / Ham API is running"}

@app.post("/predict")
def predict(request: EmailRequest):
    email_text = request.email_text.strip()

    if not email_text:
        raise HTTPException(
            status_code=400,
            detail="Email text cannot be empty",
        )

    cleaned = clean_text(email_text)
    prediction = model.predict([cleaned])[0]

    return {
        "prediction": str(prediction).upper(),
    }