# MLproject/modelling.py
import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import itertools
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys # WAJIB: Untuk membaca argumen command line

warnings.filterwarnings("ignore")

# --- 1. Fungsi Evaluasi dan Manual Logging ---
def eval_and_log_manual(model, X_test, y_test, run_id, log_model_flag=False, input_example=None):
    """Menghitung metrik dan mencatat semuanya secara manual ke MLflow."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Hitung Metrik
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    try:
        auc_roc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc_roc = 0.0

    # Log Metrik Secara Manual
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc_roc)

    # Log Confusion Matrix sebagai Artefak Visual
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix (Acc: {accuracy:.4f})")
    
    cm_path = f"confusion_matrix_{run_id}.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path, artifact_path="visual_artifacts")
    os.remove(cm_path)
    plt.close()

    # Log Model (Hanya untuk model terbaik)
    if log_model_flag:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            registered_model_name="credit_scoring_rf_manual" 
        )

    return accuracy

if __name__ == "__main__":
    
    # ------------------------------------------------------------------------
    # WAJIB: Menerima Parameter dari Command Line (sys.argv)
    # Parameter akan dilewatkan oleh MLproject: n_estimators, max_depth, dataset_path
    # ------------------------------------------------------------------------
    
    warnings.filterwarnings("ignore")
    np.random.seed(42)
    
    # Menerima argumen dan menggunakan nilai default jika argumen kosong
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    data_path = sys.argv[3] if len(sys.argv) > 3 else "dataset_preprocessing/preprocessed_data.csv"
    
    print(f"CI Run: n_estimators={n_estimators}, max_depth={max_depth}, Data Path={data_path}")

    # --- 2. Pemuatan Data ---
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"ERROR: File data preprocessing tidak ditemukan di {data_path}.")
        sys.exit(1)

    # Pisahkan Fitur (X) dan Target (y) - ASUMSI
    X = data.drop("Status_Resiko", axis=1) 
    y = data["Status_Resiko"]

    # Train-Test Split (untuk evaluasi)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2, stratify=y
    )

    # Ambil contoh input untuk log_model
    input_example = X_train.head(5)

    # --- 3. Memulai MLflow Run (Single Run, bukan loop tuning penuh) ---
    # Kita hanya menjalankan satu run representatif di CI
    
    mlflow.set_experiment("CI Workflow Resiko Kesehatan") 
    
    with mlflow.start_run(run_name=f"CI_Run_n{n_estimators}_d{max_depth}") as run:
        run_id = run.info.run_id
        
        # 4. Model Training
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42
        )
        
        # 5. Catat Parameter Secara Manual
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("data_source", data_path)
        
        # 6. Training
        model.fit(X_train, y_train)
        
        # 7. Evaluasi dan Log (Manual)
        current_accuracy = eval_and_log_manual(
            model, X_test, y_test, run_id, 
            log_model_flag=True, # Langsung simpan model ini
            input_example=input_example
        )
        
        print(f"\nCI Run Selesai. Akurasi: {current_accuracy:.4f}")