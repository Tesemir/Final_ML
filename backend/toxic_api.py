from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import os

# Get absolute path to static folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "../static")

model = joblib.load(os.path.join(BASE_DIR, "logreg_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

app = FastAPI()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    html_path = os.path.join(STATIC_DIR, "toxic_frontend.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

class CommentRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_toxicity(request: CommentRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty comment provided.")

    vec = vectorizer.transform([request.text])
    prediction = model.predict(vec)[0]
    result = {label: bool(prediction[i]) for i, label in enumerate(LABELS)}
    return {"input": request.text, "prediction": result}