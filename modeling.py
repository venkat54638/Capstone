import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os

def train_model(X_train, y_train, X_test, y_test):
    """
    Trains multiple classification models on preprocessed data and returns the results.
    """
    models = {
        'XGBoost': {
            'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'params': {'n_estimators': [100, 200], 'max_depth': [3, 6]}
        },
        'RandomForest': {
            'model': RandomForestClassifier(class_weight='balanced', random_state=42),
            'params': {'n_estimators': [100, 200], 'max_depth': [None, 10]}
        },
        'LightGBM': {
            'model': LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1),
            'params': {'n_estimators': [100, 200], 'max_depth': [3, 6]}
        },
        'LogisticRegression': {
            'model': LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42),
            'params': {'C': [0.1, 1.0, 10.0]}
        }
    }

    results = []
    best_models = {}

    for name, model_info in models.items():
        print(f"\n--- Training {name} ---")
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        # CORRECTED LINE: Reduce the number of parallel jobs to avoid memory issues.
        # Changed n_jobs=-1 to n_jobs=2. You can adjust this based on your system's RAM.
        grid = GridSearchCV(model_info['model'], model_info['params'], cv=skf, scoring='f1', n_jobs=2, verbose=1)
        
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else 'N/A',
            'log_loss': log_loss(y_test, y_proba) if y_proba is not None else 'N/A',
        }
        
        results.append({"model": name, "best_params": grid.best_params_, **metrics})
        best_models[name] = best_model

        with mlflow.start_run(run_name=f"Tuning_{name}"):
            mlflow.log_params(grid.best_params_)
            scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(scalar_metrics)
            mlflow.sklearn.log_model(best_model, "model")

    return results, best_models

# Note: The save_and_register_best_model function remains the same as in your original code.
def save_and_register_best_model(results, best_models, save_dir="saved_models"):
    """
    Identifies the best model, saves it locally, registers it in MLflow,
    and promotes it to the 'Production' stage using aliases.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if not results:
        print("No results to process. Skipping model registration.")
        return

    # Identify the best model based on F1 score
    best_model_info = sorted(results, key=lambda x: x['f1_score'], reverse=True)[0]
    best_model_name = best_model_info['model']
    best_model_obj = best_models[best_model_name]
    
    print(f"\n--- Best model identified: {best_model_name} (F1 Score: {best_model_info['f1_score']:.4f}) ---")

    # Save the single best model locally
    model_path = os.path.join(save_dir, "best_model.pkl")
    joblib.dump(best_model_obj, model_path)
    print(f"Best model '{best_model_name}' saved locally at {model_path}")

    # Log and register the best model in a new MLflow run
    print("\nRegistering the best model to MLflow Model Registry...")
    client = MlflowClient()
    
    with mlflow.start_run(run_name=f"Production_Candidate_{best_model_name}") as run:
        run_id = run.info.run_id
        
        # Log the best model as an artifact and register it simultaneously
        model_info = mlflow.sklearn.log_model(
            sk_model=best_model_obj,
            artifact_path="model",
            registered_model_name=best_model_name
        )
        
        # Log metrics and params for the final model
        best_metrics = {k: v for k, v in best_model_info.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(best_metrics)
        mlflow.log_params(best_model_info['best_params'])
        
        print(f"Model '{best_model_name}' logged and registered from run_id: {run_id}")
        
        # Use set_registered_model_alias to manage model lifecycle
        try:
            print(f"Setting alias 'production' for model version {model_info.version}...")
            client.set_registered_model_alias(
                name=best_model_name,
                alias="production",
                version=model_info.version
            )
            print(f"Model '{best_model_name}' version {model_info.version} is now aliased as 'production'.")
        except Exception as e:
            print(f"Setting model alias failed: {e}")