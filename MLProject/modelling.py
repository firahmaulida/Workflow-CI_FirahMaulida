import argparse
import json
import logging
import os
import sys
from datetime import datetime

import dagshub
import matplotlib
matplotlib.use("Agg")   # Wajib untuk server/CI
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# Konstanta
DATA_PATH       = os.path.join(os.path.dirname(__file__), "telco_churn_preprocessed.csv")
TARGET_COLUMN   = "Churn"
EXPERIMENT_NAME = "Telco_Churn_CI_FirahMaulida"
ARTIFACT_DIR    = "artifacts"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators",      type=int,   default=100)
    parser.add_argument("--max_depth",         type=int,   default=10)
    parser.add_argument("--min_samples_split", type=int,   default=2)
    parser.add_argument("--test_size",         type=float, default=0.2)
    parser.add_argument("--random_state",      type=int,   default=42)
    return parser.parse_args()

def run(args):
    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "test_size": args.test_size,
        "random_state": args.random_state,
    }

    # --- PERBAIKAN UNTUK OTOMASI CI (Agar tidak minta login browser) ---
    token = os.getenv("DAGSHUB_TOKEN")
    if token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = "firahmaulida"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        # Memberitahu DagsHub untuk menggunakan token secara langsung
        os.environ["DAGSHUB_USER_TOKEN"] = token
        logger.info("✅ Login otomatis menggunakan DAGSHUB_TOKEN.")
    
    dagshub.init(repo_owner="firahmaulida", repo_name="Eksperimen_SML_FirahMaulida", mlflow=True)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load Data
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state, stratify=y)

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=args.random_state
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)

        # Artefak
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.savefig(f"{ARTIFACT_DIR}/confusion_matrix.png")
        mlflow.log_artifact(f"{ARTIFACT_DIR}/confusion_matrix.png")
        
        mlflow.sklearn.log_model(model, "model")
        logger.info(f"🚀 Training Selesai! Accuracy: {acc}")

if __name__ == "__main__":
    args = parse_args()
    run(args)
