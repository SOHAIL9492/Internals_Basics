import os
import json
import pickle
import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
import uvicorn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load best model saved by train.py
model_path = os.path.join(MODELS_DIR, "best_model.pkl")
with open(model_path, "rb") as f:
    payload = pickle.load(f)

_model      = payload["model"]
_model_name = payload["name"]

app = FastAPI()

class ArtworkFeatures(BaseModel):
    artist_reputation_score: float = Field(..., ge=1, le=10)
    artwork_age_years:        float = Field(..., ge=1, le=200)
    medium_type_index:        int   = Field(..., ge=1, le=5)
    exhibition_count:         int   = Field(..., ge=0, le=20)

@app.get("/health")
def health():
    return {"status": "running", "model": _model_name, "version": "1.0"}

@app.post("/estimate")
def estimate(req: ArtworkFeatures):
    data = pd.DataFrame([req.model_dump()])
    pred = float(_model.predict(data)[0])
    return {"prediction": pred}

if __name__ == "__main__":
    client = TestClient(app)

    health_resp = client.get("/health").json()

    test_input = {
        "artist_reputation_score": 6.8,
        "artwork_age_years": 136,
        "medium_type_index": 2,
        "exhibition_count": 6
    }
    pred_resp = client.post("/estimate", json=test_input).json()

    output = {
        "health_endpoint": "/health",
        "predict_endpoint": "/estimate",
        "port": 8000,
        "health_response": health_resp,
        "test_input": test_input,
        "prediction": pred_resp.get("prediction")
    }

    out_path = os.path.join(RESULTS_DIR, "step2_s4.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Task 2 complete -> {out_path}")

    print("Starting API server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
