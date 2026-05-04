import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE_DIR, "data", "training_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRACKING_URI    = "sqlite:///" + os.path.join(BASE_DIR, "mlruns.db")
EXPERIMENT_NAME = "gallerypulse-auction-price-lakhs"
MODEL_NAME      = "gallerypulse-auction-price-lakhs-predictor"
FEATURES        = ["artist_reputation_score", "artwork_age_years", "medium_type_index", "exhibition_count"]
TARGET          = "auction_price_lakhs"

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    # ---- Version 1: champion (already registered by register_model.py) ----
    # Read its MAE from step3 JSON
    step3_path = os.path.join(RESULTS_DIR, "step3_s6.json")
    if not os.path.exists(step3_path):
        raise RuntimeError("Run register_model.py first to create step3_s6.json")

    with open(step3_path) as f:
        step3 = json.load(f)

    champion_version = step3["version"]           # should be 1
    champion_mae     = step3["source_metric_value"]

    # Assign "production" alias to champion (version 1)
    client.set_registered_model_alias(MODEL_NAME, "production", str(champion_version))
    print(f"Alias 'production' -> version {champion_version}")

    # ---- Version 2: challenger (Ridge, random_state=99) ----
    df = pd.read_csv(DATA_PATH)
    X  = df[FEATURES]
    y  = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

    challenger_model = Ridge()
    with mlflow.start_run(run_name="Ridge_challenger_rs99") as run:
        mlflow.set_tag("team", "ml_engineering")
        challenger_model.fit(X_train, y_train)
        y_pred          = challenger_model.predict(X_test)
        challenger_mae  = float(mean_absolute_error(y_test, y_pred))
        mlflow.log_param("random_state", 99)
        mlflow.log_metric("mae", challenger_mae)
        mlflow.sklearn.log_model(challenger_model, "model")
        challenger_run_id = run.info.run_id

    # Register as version 2
    new_model_uri = f"runs:/{challenger_run_id}/model"
    v2_result     = mlflow.register_model(new_model_uri, MODEL_NAME)
    challenger_version = int(v2_result.version)
    print(f"Challenger registered as version {challenger_version}, MAE={challenger_mae:.4f}")

    # Compare and promote if challenger is better
    if challenger_mae < champion_mae:
        client.set_registered_model_alias(MODEL_NAME, "production", str(challenger_version))
        action = "promoted"
    else:
        action = "kept"

    print(f"Action: {action}.")

    output = {
        "registered_model_name": MODEL_NAME,
        "alias_name":            "production",
        "champion_version":      champion_version,   # always version 1 (original champion)
        "challenger_version":    challenger_version,
        "action":                action
    }

    out_path = os.path.join(RESULTS_DIR, "step4_s7.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Task 4 complete -> {out_path}")

if __name__ == "__main__":
    main()
