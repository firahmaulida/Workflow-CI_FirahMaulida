"""
modelling.py  (Kriteria 3 — MLflow Project Entry Point)
=========================================================
Script training RandomForest yang dirancang sebagai entry point MLflow Project.
Menerima hyperparameter dari CLI (didefinisikan di file MLProject).
Mencatat eksperimen ke DagsHub Remote MLflow Tracking Server.

Author  : FirahMaulida
Course  : Membangun Sistem Machine Learning - Dicoding
Kriteria: 3 — Workflow CI / MLflow Project
DagsHub : https://dagshub.com/firahmaulida/Eksperimen_SML_FirahMaulida
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import dagshub
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (wajib untuk CI/server)
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

# ---------------------------------------------------------------------------
# Konfigurasi Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------
DATA_PATH       = os.path.join(os.path.dirname(__file__), "telco_churn_preprocessed.csv")
TARGET_COLUMN   = "Churn"
EXPERIMENT_NAME = "Telco_Churn_CI_FirahMaulida"
ARTIFACT_DIR    = "artifacts"


# ---------------------------------------------------------------------------
# 1. Argument Parser (menerima parameter dari MLProject)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Mem-parsing argumen CLI yang dikirim oleh MLflow Project runner.

    Returns
    -------
    argparse.Namespace
        Object berisi nilai semua hyperparameter.
    """
    parser = argparse.ArgumentParser(
        description="Training pipeline Telco Churn — MLflow Project Entry Point"
    )
    parser.add_argument("--n_estimators",      type=int,   default=100,  help="Jumlah pohon dalam hutan")
    parser.add_argument("--max_depth",         type=int,   default=10,   help="Kedalaman maksimum pohon")
    parser.add_argument("--min_samples_split", type=int,   default=2,    help="Minimum sampel untuk split node")
    parser.add_argument("--test_size",         type=float, default=0.2,  help="Proporsi data test (0.0 - 1.0)")
    parser.add_argument("--random_state",      type=int,   default=42,   help="Random seed untuk reproduktibilitas")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 2. Data Loading & Splitting
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """
    Memuat dataset preprocessed dari path yang diberikan.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Memuat dataset dari: %s", filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {filepath}")
    df = pd.read_csv(filepath)
    logger.info("Dataset dimuat — Shape: %s", df.shape)
    return df


def split_data(
    df: pd.DataFrame,
    target: str,
    test_size: float,
    random_state: int,
) -> tuple:
    """
    Memisahkan fitur & target, lalu membagi ke train/test split stratified.

    Returns
    -------
    tuple : X_train, X_test, y_train, y_test, feature_names
    """
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        "Split — Train: %d | Test: %d | Fitur: %d",
        len(X_train), len(X_test), X.shape[1],
    )
    return X_train, X_test, y_train, y_test, list(X.columns)


# ---------------------------------------------------------------------------
# 3. Model Training & Evaluation
# ---------------------------------------------------------------------------

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int,
    max_depth: int,
    min_samples_split: int,
    random_state: int,
) -> RandomForestClassifier:
    """
    Melatih RandomForestClassifier dengan parameter yang diberikan.

    Returns
    -------
    RandomForestClassifier : model yang sudah dilatih.
    """
    logger.info(
        "Training RF — n_estimators=%d | max_depth=%s | min_samples_split=%d",
        n_estimators, max_depth, min_samples_split,
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",   # Menangani class imbalance Churn
    )
    model.fit(X_train, y_train)
    logger.info("Training selesai!")
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict, np.ndarray]:
    """
    Mengevaluasi model dan menghitung semua metrik klasifikasi.

    Returns
    -------
    tuple[dict, np.ndarray] : (metrics_dict, y_pred)
    """
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy"          : round(accuracy_score(y_test, y_pred), 6),
        "f1_weighted"       : round(f1_score(y_test, y_pred, average="weighted"), 6),
        "f1_macro"          : round(f1_score(y_test, y_pred, average="macro"), 6),
        "precision_weighted": round(precision_score(y_test, y_pred, average="weighted"), 6),
        "recall_weighted"   : round(recall_score(y_test, y_pred, average="weighted"), 6),
        "roc_auc"           : round(roc_auc_score(y_test, y_pred_prob), 6),
    }
    return metrics, y_pred


# ---------------------------------------------------------------------------
# 4. Artifact Generation
# ---------------------------------------------------------------------------

def save_confusion_matrix(
    y_test: pd.Series,
    y_pred: np.ndarray,
    output_dir: str,
) -> str:
    """Membuat dan menyimpan plot Confusion Matrix ke PNG."""
    os.makedirs(output_dir, exist_ok=True)
    cm   = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    ax.set_title("Confusion Matrix — Telco Churn\nFirahMaulida", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def save_feature_importance(
    model: RandomForestClassifier,
    feature_names: list[str],
    output_dir: str,
    top_n: int = 20,
) -> str:
    """Membuat dan menyimpan plot Feature Importance top-N ke PNG."""
    os.makedirs(output_dir, exist_ok=True)
    importances  = pd.Series(model.feature_importances_, index=feature_names)
    top_features = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(9, 7))
    colors  = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_features)))
    top_features.plot(kind="barh", ax=ax, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title(f"Top {top_n} Feature Importances — FirahMaulida", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def save_classification_report_txt(
    y_test: pd.Series,
    y_pred: np.ndarray,
    metrics: dict,
    params: dict,
    output_dir: str,
) -> str:
    """Menyimpan Classification Report ke file TXT."""
    os.makedirs(output_dir, exist_ok=True)
    report    = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = (
        f"{'='*60}\n"
        f"  CLASSIFICATION REPORT — Telco Customer Churn\n"
        f"  Author    : FirahMaulida\n"
        f"  Timestamp : {timestamp}\n"
        f"{'='*60}\n\n"
        f"HYPERPARAMETERS:\n"
        + "\n".join(f"  {k:22s}: {v}" for k, v in params.items())
        + f"\n\nMETRICS:\n"
        + "\n".join(f"  {k:22s}: {v:.4f}" for k, v in metrics.items())
        + f"\n\nDETAILED REPORT:\n{report}\n"
    )
    path = os.path.join(output_dir, "classification_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Saved: %s", path)
    return path


def save_summary_json(
    metrics: dict,
    params: dict,
    run_id: str,
    output_dir: str,
) -> str:
    """Menyimpan ringkasan eksperimen ke file JSON."""
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "experiment_info": {
            "author"       : "FirahMaulida",
            "course"       : "Membangun Sistem Machine Learning - Dicoding",
            "algorithm"    : "RandomForestClassifier",
            "timestamp"    : datetime.now().isoformat(),
            "mlflow_run_id": run_id,
            "dagshub_url"  : "https://dagshub.com/firahmaulida/Eksperimen_SML_FirahMaulida",
        },
        "hyperparameters": params,
        "metrics"        : metrics,
    }
    path = os.path.join(output_dir, "run_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, default=str)
    logger.info("Saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# 5. MLflow Manual Logging
# ---------------------------------------------------------------------------

def log_to_mlflow(
    model: RandomForestClassifier,
    metrics: dict,
    params: dict,
    y_test: pd.Series,
    y_pred: np.ndarray,
    feature_names: list[str],
) -> str:
    """
    Mencatat semua parameter, metrik, artefak, dan model ke MLflow
    menggunakan Manual Logging (bukan autolog).

    Returns
    -------
    str : run_id dari MLflow aktif.
    """
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # --- Log Parameters ---
        for key, val in params.items():
            mlflow.log_param(key, val)

        # --- Log Metrics ---
        for key, val in metrics.items():
            mlflow.log_metric(key, val)

        # --- Generate & Log Artifacts ---
        cm_path   = save_confusion_matrix(y_test, y_pred, ARTIFACT_DIR)
        fi_path   = save_feature_importance(model, feature_names, ARTIFACT_DIR)
        cr_path   = save_classification_report_txt(y_test, y_pred, metrics, params, ARTIFACT_DIR)
        sum_path  = save_summary_json(metrics, params, run_id, ARTIFACT_DIR)

        mlflow.log_artifact(cm_path,  artifact_path="plots")
        mlflow.log_artifact(fi_path,  artifact_path="plots")
        mlflow.log_artifact(cr_path,  artifact_path="reports")
        mlflow.log_artifact(sum_path, artifact_path="summaries")

        # --- Log Model (untuk mlflow models build-docker) ---
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="TelcoChurn_RF_FirahMaulida",
            input_example=pd.DataFrame(
                np.zeros((1, len(feature_names))), columns=feature_names
            ),
        )

        logger.info("─" * 55)
        logger.info("MLflow Run ID : %s", run_id)
        logger.info("Accuracy      : %.4f", metrics["accuracy"])
        logger.info("F1 (Weighted) : %.4f", metrics["f1_weighted"])
        logger.info("ROC-AUC       : %.4f", metrics["roc_auc"])
        logger.info("─" * 55)

    return run_id


# ---------------------------------------------------------------------------
# 6. Pipeline Utama
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    """
    Pipeline utama yang dipanggil oleh MLflow Project runner.
    Menghubungkan ke DagsHub, melatih model, dan mencatat semua hasil.

    Parameters
    ----------
    args : argparse.Namespace
        Argumen CLI yang sudah di-parse.
    """
    logger.info("=" * 60)
    logger.info("  WORKFLOW CI — TELCO CUSTOMER CHURN")
    logger.info("  Author   : FirahMaulida | Dicoding MSML")
    logger.info("  Kriteria : 3 — MLflow Project + CI")
    logger.info("=" * 60)

    params = {
        "n_estimators"    : args.n_estimators,
        "max_depth"       : args.max_depth,
        "min_samples_split": args.min_samples_split,
        "test_size"       : args.test_size,
        "random_state"    : args.random_state,
        "algorithm"       : "RandomForestClassifier",
        "class_weight"    : "balanced",
    }
    logger.info("Parameters: %s", params)

    # --- Inisialisasi DagsHub ---
    logger.info("Menghubungkan ke DagsHub MLflow...")
    dagshub.init(
        repo_owner="firahmaulida",
        repo_name="Eksperimen_SML_FirahMaulida",
        mlflow=True,
    )
    mlflow.set_experiment(EXPERIMENT_NAME)

    # --- Data ---
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, feature_names = split_data(
        df, TARGET_COLUMN, args.test_size, args.random_state
    )

    # --- Train ---
    model = train_model(
        X_train, y_train,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=args.random_state,
    )

    # --- Evaluate ---
    metrics, y_pred = evaluate_model(model, X_test, y_test)

    # --- Log to MLflow / DagsHub ---
    run_id = log_to_mlflow(model, metrics, params, y_test, y_pred, feature_names)

    logger.info("=" * 60)
    logger.info("  PIPELINE SELESAI!")
    logger.info("  Run ID  : %s", run_id)
    logger.info("  DagsHub : https://dagshub.com/firahmaulida/Eksperimen_SML_FirahMaulida.mlflow")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    run(args)
