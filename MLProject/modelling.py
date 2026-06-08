import argparse
import logging
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# Wajib untuk server tanpa layar
matplotlib.use("Agg") 
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

def run():
    # 1. AMBIL ARGUMEN
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    # 2. SETTING DAGSHUB (Wajib Paling Atas)
    token = os.getenv("DAGSHUB_TOKEN")
    if token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = "firahmaulida"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        mlflow.set_tracking_uri("https://dagshub.com/firahmaulida/Eksperimen_SML_FirahMaulida.mlflow")
        # MLflow akan otomatis membuat experiment jika belum ada
        mlflow.set_experiment("Telco_Churn_CI_FirahMaulida")

    # 3. LOADING DATA
    df = pd.read_csv("telco_churn_preprocessed.csv")
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    # 4. PROSES TRAINING (Tanpa start_run tambahan agar tidak bentrok)
    logger.info("Memulai pelatihan model...")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators, 
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=args.random_state
    )
    model.fit(X_train, y_train)
    
    # Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # LOGGING MANUAL (Bintang 5)
    # MLflow otomatis akan mencatat ini ke Run yang dibuat oleh 'mlflow run'
    mlflow.log_params(vars(args))
    mlflow.log_metric("accuracy", acc)
    
    # Simpan Artefak
    os.makedirs("artifacts", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
    fig.savefig("artifacts/cm.png")
    mlflow.log_artifact("artifacts/cm.png")
    
    # Simpan Model
    mlflow.sklearn.log_model(model, "model")
    logger.info(f"✅ SUCCESS! Accuracy: {acc}")

if __name__ == "__main__":
    run()
