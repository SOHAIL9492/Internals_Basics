import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import mlflow
from mlflow.tracking import MlflowClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'training_data.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

def main():
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    experiment_name = "gallerypulse-auction-price-lakhs"
    mlflow.set_experiment(experiment_name)
    
    model_name = "gallerypulse-auction-price-lakhs-predictor"
    client = MlflowClient()

    try:
        client.set_registered_model_alias(model_name, "production", "1")
        v1_model = client.get_model_version_by_alias(model_name, "production")
    except Exception as e:
        print(f"Failed to fetch model version 1: {e}")
        return
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    features = ['artist_reputation_score', 'artwork_age_years', 'medium_type_index', 'exhibition_count']
    X = df[features]
    y = df['auction_price_lakhs']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)
    
    model = Ridge()
    with mlflow.start_run(run_name="Ridge_rs99") as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(model, "model")
        
        new_run_id = run.info.run_id

    model_uri = f"runs:/{new_run_id}/model"
    new_version_result = mlflow.register_model(model_uri, model_name)
    new_version = new_version_result.version
    
    v1_run = client.get_run(v1_model.run_id)
    v1_mae = v1_run.data.metrics["mae"]
    
    if mae < v1_mae:
        client.set_registered_model_alias(model_name, "production", new_version)
        action = "promoted"
    else:
        action = "kept"

    output_data = {
        "registered_model_name": model_name,
        "alias_name": "production",
        "champion_version": int(new_version) if action == "promoted" else 1,
        "challenger_version": int(new_version),
        "action": action
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'step4_s7.json'), 'w') as f:
        json.dump(output_data, f, indent=4)
    print("Task 4 Complete. JSON saved to results/step4_s7.json")

if __name__ == "__main__":
    main()
