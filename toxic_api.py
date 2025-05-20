from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib

# Load model and vectorizer
model = joblib.load("logreg_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

app = FastAPI()

# Serve static files (HTML/JS/CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Serve the frontend at /
@app.get("/", response_class=HTMLResponse)
async def serve_home():
    with open("static/toxic_frontend.html", "r", encoding="utf-8") as f:
        return f.read()

# Model input type
class CommentRequest(BaseModel):
    text: str

# Prediction endpoint
@app.post("/predict")
def predict_toxicity(request: CommentRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty comment provided.")

    vec = vectorizer.transform([request.text])
    prediction = model.predict(vec)[0]
    result = {label: bool(prediction[i]) for i, label in enumerate(LABELS)}
    return {"input": request.text, "prediction": result}