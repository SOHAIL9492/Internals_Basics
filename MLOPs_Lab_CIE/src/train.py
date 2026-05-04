import os
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

TRACKING_URI = "sqlite:///" + os.path.join(BASE_DIR, "mlruns.db")
EXPERIMENT_NAME = "gallerypulse-auction-price-lakhs"
FEATURES = ["artist_reputation_score", "artwork_age_years", "medium_type_index", "exhibition_count"]
TARGET = "auction_price_lakhs"

def train_and_log(name, model, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("team", "ml_engineering")
        # Log hyperparameters
        params = model.get_params()
        for k, v in params.items():
            mlflow.log_param(k, v)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae  = float(mean_absolute_error(y_test, y_pred))
        rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)
        r2   = float(r2_score(y_test, y_pred))

        mlflow.log_metric("mae",  mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2",   r2)
        mlflow.sklearn.log_model(model, "model")

        run_id = run.info.run_id
    return {"name": name, "mae": mae, "rmse": rmse, "r2": r2, "run_id": run_id, "model": model}

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []
    results.append(train_and_log("LinearRegression", LinearRegression(), X_train, X_test, y_train, y_test))
    results.append(train_and_log("Ridge",            Ridge(),            X_train, X_test, y_train, y_test))

    best = min(results, key=lambda x: x["mae"])

    # Save best model as pickle for API use
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": best["model"], "name": best["name"]}, f)
    print("Best model '" + best['name'] + "' saved to " + model_path)

    output = {
        "experiment_name": EXPERIMENT_NAME,
        "models": [
            {"name": r["name"], "mae": r["mae"], "rmse": r["rmse"], "r2": r["r2"]}
            for r in results
        ],
        "best_model": best["name"],
        "best_metric_name": "mae",
        "best_metric_value": best["mae"]
    }

    out_path = os.path.join(RESULTS_DIR, "step1_s1.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Task 1 complete -> {out_path}")

if __name__ == "__main__":
    main()
