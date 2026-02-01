# Path: main.py
import sys
import subprocess
import time
import os
import pandas as pd

# Import our custom engines
from src.data_engine import generate_churn_data
from src.feature_engineering import preprocess_data, split_data
from src.model_trainer import train_churn_model, evaluate_model
# NEW: Import the visualizer
from src.visualizer import generate_all_plots

def run_pipeline():
    print("="*50)
    print(">>> STARTING TELECOM CHURN PREDICTION PIPELINE")
    print("="*50)

    # --- STEP 1: Data Generation ---
    print("\n[Step 1] Generating Synthetic Customer Data...")
    raw_path = 'data/raw/telecom_churn_v1.csv'
    
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    df = generate_churn_data(num_customers=2000, output_path=raw_path)
    time.sleep(1)

    # --- STEP 1.5: Visualization (NEW STEP) ---
    print("\n[Step 1.5] Generating Strategic Reports...")
    generate_all_plots(df)
    time.sleep(1)

    # --- STEP 2: Feature Engineering & Processing ---
    print("\n[Step 2] Processing Data for Machine Learning...")
    
    X, y = preprocess_data(df)
    
    # Save Processed Data
    processed_path = 'data/processed/churn_processed_ready.csv'
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    
    df_processed = X.copy()
    df_processed['Churn'] = y
    df_processed.to_csv(processed_path, index=False)
    print(f"[INFO] Cleaned data saved to: {processed_path}")

    # Split Data
    X_train, X_test, y_train, y_test = split_data(X, y)
    time.sleep(1)

    # --- STEP 3: Model Training & Evaluation ---
    print("\n[Step 3] Training Random Forest Classifier...")
    
    model = train_churn_model(X_train, y_train)
    
    print("\n[INFO] Evaluating Model Performance...")
    acc = evaluate_model(model, X_test, y_test)
    
    print(f"[RESULT] Final Model Accuracy: {acc*100:.2f}%")
    time.sleep(1)

    # --- STEP 4: Launch Dashboard ---
    print("\n[Step 4] Launching Interactive Intelligence Dashboard...")
    print("Opening browser in 3 seconds...")
    time.sleep(2)
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard.py"])

if __name__ == "__main__":
    run_pipeline()