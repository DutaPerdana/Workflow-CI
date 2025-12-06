# MLproject/modelling.py
import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys 

warnings.filterwarnings("ignore")

# --- Fungsi Evaluasi dan Manual Logging ---
def eval_and_log_manual(model, X_test, y_test, run_id, input_example=None):
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

    # Log Model Terbaik
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example,
        registered_model_name="credit_scoring_ci_model" 
    )
    return accuracy

if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    np.random.seed(42)
    
    # ------------------------------------------------------------------------
    # Menerima Parameter dari Command Line (sys.argv)
    # ------------------------------------------------------------------------
    
    if len(sys.argv) < 4:
        # Ini hanya terjadi jika MLproject gagal, dan script dijalankan secara manual tanpa argumen
        print("FATAL ERROR: Jumlah argumen tidak sesuai (Membutuhkan 3 argumen: n_estimators, max_depth, data_path).")
        sys.exit(1)
        
    # Mengambil nilai yang dilewatkan dari MLproject
    n_estimators = int(sys.argv[1])
    max_depth = int(sys.argv[2])
    data_path = sys.argv[3] 
    
    # NOTE: Kita tidak perlu set_tracking_uri karena MLflow run akan melacak ke ./mlruns
    mlflow.set_experiment("CI Workflow Credit Scoring") 
    
    print(f"CI Run Parameters: n_estimators={n_estimators}, max_depth={max_depth}, Data Path={data_path} (CWD: {os.getcwd()})")

    # --- 2. Pemuatan Data ---
    try:
        # Jalur data sekarang seharusnya sudah benar relatif terhadap CWD (MLproject/)
        data = pd.read_csv(data_path) 
    except FileNotFoundError:
        print(f"ERROR FATAL: File data preprocessing tidak ditemukan di {data_path}. Gagal Memuat.")
        sys.exit(1)
        
    # Pisahkan Fitur (X) dan Target (y) - ASUMSI
    X = data.drop("Status_Resiko", axis=1) 
    y = data["Status_Resiko"]

    # Train-Test Split (untuk evaluasi)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2, stratify=y
    )
    input_example = X_train.head(5)

    # --- 3. Memulai MLflow Run (Single Run CI) ---
    # TIDAK MENGGUNAKAN 'with mlflow.start_run()' karena sudah dimulai oleh mlflow run
    
    # Ambil Run ID yang sudah aktif
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        
        mlflow.set_tag("mlflow.runName", f"CI_n{n_estimators}_d{max_depth}")
        
        # Log Parameter Secara Manual
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("data_source", data_path)
        
        # Model Training
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluasi dan Log Artefak Model
        current_accuracy = eval_and_log_manual(
            model, X_test, y_test, run_id, 
            input_example=input_example
        )
        
        print(f"\nCI Run Selesai. Akurasi: {current_accuracy:.4f}")