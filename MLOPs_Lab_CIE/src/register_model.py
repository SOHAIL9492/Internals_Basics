import os
import json
import mlflow

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

mlflow.set_tracking_uri("sqlite:///mlruns.db")

def main():
    experiment_name = "gallerypulse-auction-price-lakhs"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print("Experiment not found. Run train.py first.")
        return

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.mae ASC"]
    )
    if runs.empty:
        print("No runs found.")
        return
        
    best_run = runs.iloc[0]
    best_run_id = best_run.run_id
    best_mae = best_run["metrics.mae"]

    model_name = "gallerypulse-auction-price-lakhs-predictor"
    model_uri = f"runs:/{best_run_id}/model"

    result = mlflow.register_model(model_uri, model_name)
    
    output_data = {
        "registered_model_name": model_name,
        "version": int(result.version),
        "run_id": best_run_id,
        "source_metric": "mae",
        "source_metric_value": float(best_mae)
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'step3_s6.json'), 'w') as f:
        json.dump(output_data, f, indent=4)
    print("Task 3 Complete. JSON saved to results/step3_s6.json")

if __name__ == "__main__":
    main()
