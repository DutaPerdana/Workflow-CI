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
# Fungsi ini mencakup semua metrik yang dicakup oleh autolog (dan lebih)
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

    # Log Metrik Secara Manual (WAJIB SKILLED)
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
    
    if len(sys.argv) < 3: # Hanya butuh n_estimators dan max_depth
        print("FATAL ERROR: Jumlah argumen tidak sesuai (Membutuhkan n_estimators dan max_depth).")
        sys.exit(1)
        
    # Mengambil nilai yang dilewatkan dari MLproject
    n_estimators = int(sys.argv[1])
    max_depth = int(sys.argv[2])
    
    # ------------------------------------------------------------------------
    # WAJIB FIX PATH: Membangun jalur file secara absolut
    # ------------------------------------------------------------------------
    
    # GITHUB_ROOT adalah root repository di runner CI
    GITHUB_ROOT = os.environ.get('GITHUB_WORKSPACE', os.getcwd()) 
    
    # Lokasi file data relatif terhadap root repository (Workflow-CI/)
    RELATIVE_DATA_PATH = "MLproject/dataset_preprocessing/preprocessed_data.csv"
    
    # Buat path absolut
    file_path = os.path.join(GITHUB_ROOT, RELATIVE_DATA_PATH)
    
    print(f"CI Run Parameters: n_estimators={n_estimators}, max_depth={max_depth}, Data Path={file_path}")

    # --- 2. Pemuatan Data ---
    try:
        data = pd.read_csv(file_path) 
    except FileNotFoundError:
        print(f"ERROR FATAL: File data preprocessing tidak ditemukan di {file_path}. Gagal Memuat.")
        sys.exit(1)
        
    # Pisahkan Fitur (X) dan Target (y) - PERBAIKI NAMA KOLOM TARGET
    X_train, X_test, y_train, y_test = train_test_split(
        # Drop kolom target yang BENAR: Status_Resiko
        data.drop("Status_Resiko", axis=1), 
        # Target yang BENAR: Status_Resiko
        data["Status_Resiko"], 
        random_state=42,
        test_size=0.2
    )
    input_example = X_train.head(5)

    # --- 3. Memulai MLflow Run (Single Run CI) ---
    mlflow.set_experiment("CI Workflow Credit Scoring") 
    
    # Log Run
    with mlflow.start_run(run_name=f"CI_n{n_estimators}_d{max_depth}") as run:
        run_id = run.info.run_id
        
        # Log Parameter Secara Manual
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("data_source", RELATIVE_DATA_PATH)
        
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