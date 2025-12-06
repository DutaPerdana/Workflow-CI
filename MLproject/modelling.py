# MLproject/modelling.py
import argparse
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import numpy as np
import time
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import sys

warnings.filterwarnings("ignore")

# --- 1. Fungsi Evaluasi dan Manual Logging (Dipersingkat untuk Contoh) ---
def eval_and_log_manual(model, X_test, y_test, run_id, input_example=None):
    # Logika Metrik Lengkap dari Kriteria 2 Skilled
    # ... (Hitung metrik, buat confusion matrix, log metrik/artefak)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # [Tambahkan semua logika log metrik/visualisasi Kriteria 2 Skilled di sini]

    # Log Model Terbaik
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example,
        registered_model_name="credit_scoring_ci_model" 
    )
    return accuracy

# ============================================================
# 2. Ambil argumen dari Command Line menggunakan argparse
# ============================================================
parser = argparse.ArgumentParser()
# Kita akan meneruskan path relatif dari ROOT REPOSITORY
parser.add_argument("--data_path", type=str, required=True, default="MLproject/dataset_preprocessing/preprocessed_data.csv") 
parser.add_argument("--n_estimators", type=int, default=300)
parser.add_argument("--max_depth", type=int, default=15)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# ============================================================
# 3. MLflow config — FIX TRACKING KE FILE URI (LOKAL)
# ============================================================
# Menggunakan jalur relatif dari lokasi script untuk mlruns/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Catatan: mlruns akan dibuat di root repository karena kita memanggil python dari root
TRACKING_DIR = "mlruns" 

mlflow.set_tracking_uri("file://" + os.path.join(os.getcwd(), TRACKING_DIR)) 
mlflow.set_experiment("CI_Workflow_Resiko_Kesehatan")


# ============================================================
# 4. Load Data (Fix Path)
# ============================================================

# args.data_path akan menjadi MLProject/dataset_preprocessing/preprocessed_data.csv
try:
    df = pd.read_csv(args.data_path) 
except FileNotFoundError:
    print(f"ERROR FATAL: File data preprocessing tidak ditemukan di {args.data_path}. Gagal Memuat.")
    sys.exit(1)

# Target dan Split Data (Sesuai Koreksi Kriteria 2: Status_Resiko)
X = df.drop("Status_Resiko", axis=1)
y = df["Status_Resiko"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_state
)
input_example = X_train.head(5)

# ============================================================
# 5. Training dan Logging
# ============================================================
with mlflow.start_run(run_name=f"CI_n{args.n_estimators}_d{args.max_depth}") as run:
    run_id = run.info.run_id
    
    # Model Training
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )

    start = time.time()
    model.fit(X_train, y_train)
    inference_time = time.time() - start

    # Log Parameter Manual
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("data_path", args.data_path)

    # Log Metrics (Gantikan dengan logika Kriteria 2 Anda)
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # Log Model dan Artefak Visual (dengan run_id)
    eval_and_log_manual(model, X_test, y_test, run_id, input_example) 


print("Training CI selesai. Akurasi =", accuracy)