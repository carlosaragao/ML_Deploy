import os
import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .preprocessing import preprocess_image

app = FastAPI(title="MNIST Predictor (Joblib)")

# CORS (opcional, ajuda se você abrir um HTML separado no browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Caminho absoluto para o artefato (robusto independente de onde você rodar)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "training", "artifacts", "model.joblib"))

pipeline = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # validação mínima
    if not file.content_type or not file.content_type.startswith("image/"):
        return {"error": "Envie um arquivo de imagem (png/jpg/webp/etc)."}

    X = preprocess_image(file.file)

    # Probabilidades (LogisticRegression suporta predict_proba)
    probs = pipeline.predict_proba(X)[0]
    digit = int(np.argmax(probs))
    confidence = float(np.max(probs))

    # Top-3 opções (legal pra demo)
    top3_idx = probs.argsort()[-3:][::-1]
    top3 = [{"digit": int(i), "prob": float(probs[i])} for i in top3_idx]

    return {
        "digit": digit,
        "confidence": round(confidence, 4),
        "top3": top3
    }