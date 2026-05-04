import os
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import pandas as pd
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

mlflow.set_tracking_uri("sqlite:///mlruns.db")

app = FastAPI()

try:
    experiment_name = "gallerypulse-auction-price-lakhs"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.mae ASC"]
        )
        if not runs.empty:
            best_run = runs.iloc[0]
            best_run_id = best_run.run_id
            model_uri = f"runs:/{best_run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
            best_model_name = best_run.get("params.model", "Unknown")
        else:
            model = None
            best_model_name = "None"
    else:
        model = None
        best_model_name = "None"
except Exception as e:
    model = None
    best_model_name = str(e)

class PredictRequest(BaseModel):
    artist_reputation_score: float = Field(..., ge=1, le=10)
    artwork_age_years: float = Field(..., ge=1, le=200)
    medium_type_index: int = Field(..., ge=1, le=5)
    exhibition_count: int = Field(..., ge=0, le=20)

@app.get("/health")
def health():
    return {
        "status": "running",
        "model": best_model_name,
        "version": "1.0"
    }

@app.post("/estimate")
def estimate(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please run train.py first.")
    data = pd.DataFrame([req.model_dump()])
    try:
        prediction = model.predict(data)[0]
        return {"prediction": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Generate the required JSON via TestClient
    client = TestClient(app)
    
    health_response = client.get("/health").json()
    
    test_input = {
        "artist_reputation_score": 6.8,
        "artwork_age_years": 136,
        "medium_type_index": 2,
        "exhibition_count": 6
    }
    
    predict_response = client.post("/estimate", json=test_input).json()
    
    output_data = {
        "health_endpoint": "/health",
        "predict_endpoint": "/estimate",
        "port": 8000,
        "health_response": health_response,
        "test_input": test_input,
        "prediction": predict_response.get("prediction")
    }
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'step2_s4.json'), 'w') as f:
        json.dump(output_data, f, indent=4)
    print("Task 2 Complete. JSON saved to results/step2_s4.json")
    
    print("Starting server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
