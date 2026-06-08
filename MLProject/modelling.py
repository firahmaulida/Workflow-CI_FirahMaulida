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

matplotlib.use("Agg") 
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    # 1. SET TRACKING URI (Wajib di awal)
    token = os.getenv("DAGSHUB_TOKEN")
    if token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = "firahmaulida"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        mlflow.set_tracking_uri("https://dagshub.com/firahmaulida/Eksperimen_SML_FirahMaulida.mlflow")
    
    mlflow.set_experiment("Telco_Churn_CI_FirahMaulida")

    # 2. LOADING DATA
    df = pd.read_csv("telco_churn_preprocessed.csv")
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    # 3. LOGGING LOGIC (KUNCI AGAR TIDAK ERROR)
    # Jika dijalankan lewat 'mlflow run', sudah ada run yang aktif. 
    # Jika tidak ada, baru kita buat baru.
    active_run = mlflow.active_run()
    if active_run:
        logger.info(f"Menggunakan run yang sudah aktif: {active_run.info.run_id}")
        execute_training(X_train, X_test, y_train, y_test, args)
    else:
        with mlflow.start_run(run_name="Manual_Run"):
            execute_training(X_train, X_test, y_train, y_test, args)

def execute_training(X_train, X_test, y_train, y_test, args):
    model = RandomForestClassifier(
        n_estimators=args.n_estimators, 
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=args.random_state
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    
    # Log params & metrics secara manual (Bintang 5)
    mlflow.log_params(vars(args))
    mlflow.log_metric("accuracy", acc)
    
    # Simpan Artefak
    os.makedirs("artifacts", exist_ok=True)
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
    fig.savefig("artifacts/cm.png")
    mlflow.log_artifact("artifacts/cm.png")
    
    mlflow.sklearn.log_model(model, "model")
    logger.info(f"✅ SUCCESS! Accuracy: {acc}")

if __name__ == "__main__":
    run()
