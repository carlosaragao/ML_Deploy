import json
import os
import time
from datetime import UTC, datetime

import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

print("📥 Carregando MNIST...")
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

# normaliza pixels
X = X / 255.0
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000
    ))
])

print("🧠 Treinando modelo...")
start = time.perf_counter()

pipeline.fit(X_train, y_train)

elapsed = time.perf_counter() - start
print(f"⏱️ Tempo de treino: {elapsed:.2f} segundos")

y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy: {acc:.4f}")

# salvar artefato
model_path = os.path.join(ARTIFACT_DIR, "model.joblib")
joblib.dump(pipeline, model_path)

# metadata
metadata = {
    "model": "LogisticRegression",
    "dataset": "MNIST",
    "input_shape": [28, 28],
    "accuracy": acc,
    "trained_at": datetime.now(UTC).isoformat()
}

with open(os.path.join(ARTIFACT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("📦 Artefatos salvos com sucesso.")