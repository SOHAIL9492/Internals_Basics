import os
import json
import mlflow

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRACKING_URI    = "sqlite:///" + os.path.join(BASE_DIR, "mlruns.db")
EXPERIMENT_NAME = "gallerypulse-auction-price-lakhs"
MODEL_NAME      = "gallerypulse-auction-price-lakhs-predictor"

def main():
    mlflow.set_tracking_uri(TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        raise RuntimeError("Experiment not found. Run train.py first.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.mae ASC"]
    )
    if runs.empty:
        raise RuntimeError("No runs found.")

    best_run    = runs.iloc[0]
    best_run_id = best_run["run_id"]
    best_mae    = float(best_run["metrics.mae"])

    model_uri = f"runs:/{best_run_id}/model"
    result    = mlflow.register_model(model_uri, MODEL_NAME)
    version   = int(result.version)

    output = {
        "registered_model_name": MODEL_NAME,
        "version": version,
        "run_id": best_run_id,
        "source_metric": "mae",
        "source_metric_value": best_mae
    }

    out_path = os.path.join(RESULTS_DIR, "step3_s6.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Task 3 complete -> {out_path}")

if __name__ == "__main__":
    main()
