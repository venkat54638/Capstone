from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import logging
import json
import os

# It's good practice to use pyngrok for local development/SageMaker notebooks
# but you might remove it for a different deployment environment.


app = Flask(__name__)

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Model, Preprocessor, and Expected Columns at Startup ---
try:
    # Load the pre-fitted scikit-learn pipeline
    preprocessor = joblib.load("preprocessor.joblib")
    logger.info("Preprocessor loaded successfully.")
    
    # Load the trained model
    model = joblib.load("saved_models/best_model.pkl")
    logger.info("Model loaded successfully.")
    
    # Load the list of column names the model was trained on
    with open('model_columns.json', 'r') as f:
        model_columns = json.load(f)
    logger.info("Model columns loaded successfully.")

except FileNotFoundError as e:
    logger.error(f"A required file was not found: {e}. Make sure 'preprocessor.joblib', 'best_model.pkl', and 'model_columns.json' are in the correct directories.")
    preprocessor = model = model_columns = None
except Exception as e:
    logger.error(f"An error occurred during initialization: {e}")
    preprocessor = model = model_columns = None

# --- Utility Function to Clean Column Names ---
def clean_col_names(df):
    """Converts DataFrame column names to a consistent snake_case format."""
    cols = df.columns
    new_cols = [col.strip().lower().replace(' ', '_').replace('(', '').replace(')', '') for col in cols]
    df.columns = new_cols
    return df

@app.route("/")
def home():
    """Renders the main page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handles the CSV file upload and returns predictions as JSON."""
    if not all([preprocessor, model, model_columns]):
        return jsonify({"error": "Server is not configured correctly. Model assets are missing."}), 500

    if 'file' not in request.files or not request.files['file'].filename:
        return jsonify({"error": "No file selected. Please upload a CSV file."}), 400
    
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Invalid file type. Please upload a CSV file."}), 400
    
    try:
        # Read and clean the column names of the uploaded data
        df_uploaded = pd.read_csv(file)
        df_cleaned = clean_col_names(df_uploaded.copy())

        # --- Data Validation and Alignment ---
        # Ensure all required columns are present, adding any missing ones with NaN
        missing_cols = set(model_columns) - set(df_cleaned.columns)
        if missing_cols:
            logger.warning(f"Uploaded CSV is missing columns: {list(missing_cols)}. They will be treated as missing values.")
            for c in missing_cols:
                df_cleaned[c] = np.nan
        
        # Reorder columns to match the exact order the model was trained on
        df_aligned = df_cleaned[model_columns]

        # Apply the preprocessor
        preprocessed_data = preprocessor.transform(df_aligned)

        # Make predictions and get probabilities
        predictions = model.predict(preprocessed_data)
        probabilities = model.predict_proba(preprocessed_data)[:, 1] # Probability of 'Convert'

        # Add results to the original DataFrame for a user-friendly output
        df_uploaded['Prediction'] = ['Convert' if p == 1 else 'Not Convert' for p in predictions]
        df_uploaded['Confidence'] = [f"{p:.2%}" for p in probabilities]

        # --- CORRECTION: Replace NaN with None for valid JSON conversion ---
        # This prevents the "Unexpected token 'N'" error in the browser.
        df_for_json = df_uploaded.replace({np.nan: None})
        
        # Prepare data for JSON response (for displaying in the web UI)
        results_json = df_for_json.to_dict(orient='records')
        
        return jsonify({"data": results_json})

    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    # Use a common high port like 8080, which is often open in cloud environments
    port = int(os.environ.get("PORT", 8080))
    
    

    # Run Flask app
    app.run(host="0.0.0.0", port=port, debug=True)
