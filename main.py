import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
import joblib
import warnings
import os
import numpy as np
import json

# Import the classes and functions from your other project files
# Ensure these files (data_preparation.py, ml_preprocessing.py, etc.) are in the same directory
from data_preparation import load_data, DataCleaner
from ml_preprocessing import Preprocessor
from modeling import train_model, save_and_register_best_model
from monitoring import generate_evidently_report

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

# Set the max CPU count for joblib's backend to prevent memory overload
# This is a preventative measure related to the error you faced.
os.environ["LOKY_MAX_CPU_COUNT"] = "2"

# --- Configuration for Artifacts and Directories ---
ARTIFACT_DIR = "/opt/airflow/mlartifacts"
DRIFT_DIR = "/opt/airflow/drift_reports"
MODEL_DIR = "/opt/airflow/saved_models"

# Create directories if they don't exist
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(DRIFT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    """
    Runs the end-to-end lead scoring pipeline: data loading, cleaning,
    preprocessing, model training, evaluation, and registration.
    """
    # Set the MLflow tracking URI and experiment name
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Lead Conversion Classification")
    print("Starting lead scoring pipeline...")

    # Step 1: Load Data
    df = load_data('Lead Scoring.csv')
    if df is None:
        print("Failed to load data. Exiting.")
        return

    # Step 2: Split Data into Training and Testing sets FIRST to prevent data leakage
    print("\nSplitting data into training and testing sets...")
    X = df.drop('Converted', axis=1)
    y = df['Converted']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Combine train features and target for the cleaning step
    train_df = pd.concat([X_train, y_train], axis=1)

    # Step 3: Clean and Prepare Data
    print("\nCleaning and preparing data...")
    cleaner = DataCleaner()
    # Fit the cleaner on the TRAINING data ONLY
    cleaner.fit(train_df)
    
    # Transform both training and testing data using the fitted cleaner
    train_df_cleaned = cleaner.transform(train_df)
    test_df_cleaned = cleaner.transform(X_test)

    # Separate features and target again after cleaning
    X_train_cleaned = train_df_cleaned.drop('converted', axis=1)
    y_train_cleaned = train_df_cleaned['converted']
    X_test_cleaned = test_df_cleaned
    
    print("Data preparation completed.")

    # Step 4: Preprocess Features
    print("\nPreprocessing features for modeling...")
    preprocessor = Preprocessor()
    # Fit the preprocessor on the CLEANED TRAINING data ONLY
    preprocessor.fit(X_train_cleaned, y_train_cleaned)

    # Transform both training and testing data using the fitted preprocessor
    X_train_processed = preprocessor.transform(X_train_cleaned)
    X_test_processed = preprocessor.transform(X_test_cleaned)
    
    # --- UPDATE: Save the fitted preprocessor AND the list of model columns ---
    joblib.dump(preprocessor, 'preprocessor.joblib')
    print("Fitted preprocessor saved as 'preprocessor.joblib'")
    
    # Define and save the final list of columns the model expects
    final_model_columns = preprocessor.numerical_features + preprocessor.categorical_features
    with open('model_columns.json', 'w') as f:
        json.dump(final_model_columns, f)
    print(f"Model columns saved at: model_columns.json")

    # Step 5: Train and Evaluate Models
    print("\nTraining and evaluating all candidate models...")
    try:
        results, best_models = train_model(X_train_processed, y_train_cleaned, X_test_processed, y_test)
    except Exception as e:
        print(f"\nAn error occurred during model training: {e}")
        print("This is often caused by memory issues. Try reducing 'n_jobs' in GridSearchCV within modeling.py,")
        print("or increase your system's virtual memory (paging file).")
        return # Exit the pipeline gracefully
        
    # Display a summary of all models' performance
    print("\n--- All Model Performance Summary ---")
    for result in results:
        # Print the model name as a header
        print(f"\n  Model: {result['model']}")
        # Iterate through all metrics in the result dictionary
        for metric, value in result.items():
            # Skip the 'model' key since we already printed it
            if metric == 'model':
                continue
            
            # Format numeric values to 4 decimal places for readability
            if isinstance(value, (int, float)):
                print(f"    - {metric.replace('_', ' ').title()}: {value:.4f}")
            else:
                # Print non-numeric values (like best_params or error messages) as is
                print(f"    - {metric.replace('_', ' ').title()}: {value}")

    # Step 6: Save, Register, and Promote the Best Model to Production
    save_and_register_best_model(results, best_models)
        
    # Step 7: Generate a data drift report comparing the train and test sets
    print("\nGenerating Evidently report for data drift monitoring...")
    y_test_renamed = y_test.rename('converted')
    current_data_for_report = pd.concat([X_test_cleaned, y_test_renamed], axis=1)
    generate_evidently_report(train_df_cleaned, current_data_for_report)

if __name__ == "__main__":
    main()