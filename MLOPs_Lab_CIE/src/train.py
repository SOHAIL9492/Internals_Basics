import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'training_data.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    experiment_name = "gallerypulse-auction-price-lakhs"
    mlflow.set_experiment(experiment_name)
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
        
    df = pd.read_csv(DATA_PATH)
    features = ['artist_reputation_score', 'artwork_age_years', 'medium_type_index', 'exhibition_count']
    X = df[features]
    y = df['auction_price_lakhs']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge()
    }
    
    results = []
    best_model_name = None
    best_mae = float('inf')
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("team", "ml_engineering")
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred) ** 0.5
            r2 = r2_score(y_test, y_pred)
            
            mlflow.log_param("model", name)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            
            mlflow.sklearn.log_model(model, "model")
            
            results.append({
                "name": name,
                "mae": mae,
                "rmse": rmse,
                "r2": r2
            })
            
            if mae < best_mae:
                best_mae = mae
                best_model_name = name

    output_data = {
        "experiment_name": experiment_name,
        "models": results,
        "best_model": best_model_name,
        "best_metric_name": "mae",
        "best_metric_value": best_mae
    }
    
    with open(os.path.join(RESULTS_DIR, 'step1_s1.json'), 'w') as f:
        json.dump(output_data, f, indent=4)
    print("Task 1 Complete. JSON saved to results/step1_s1.json")

if __name__ == "__main__":
    main()
