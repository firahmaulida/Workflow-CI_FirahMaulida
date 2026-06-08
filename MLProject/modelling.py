import argparse
import logging
import os
import sys
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
    # --- 1. SET UP ARGUMENTS (WAJIB SAMA DENGAN FILE MLProject) ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    # --- 2. KUNCI ANTI LOGIN (UNTUK GITHUB ACTIONS) ---
    token = os.getenv("DAGSHUB_TOKEN")
    if token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        mlflow.set_tracking_uri("https://dagshub.com/firahmaulida/Eksperimen_SML_FirahMaulida.mlflow")
        logger.info("✅ CI Mode: Token detected.")
    else:
        import dagshub
        dagshub.init(repo_owner="firahmaulida", repo_name="Eksperimen_SML_FirahMaulida", mlflow=True)
    
    mlflow.set_experiment("Telco_Churn_CI_FirahMaulida")

    # --- 3. LOADING DATA ---
    data_path = "telco_churn_preprocessed.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File {data_path} tidak ditemukan di folder MLProject!")
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    # --- 4. TRAINING & LOGGING ---
    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=args.n_estimators, 
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=args.random_state
        )
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        
        # Log all params
        mlflow.log_params(vars(args))
        mlflow.log_metric("accuracy", acc)
        
        # Simpan Artefak
        os.makedirs("artifacts", exist_ok=True)
        cm = confusion_matrix(y_test, model.predict(X_test))
        ConfusionMatrixDisplay(confusion_matrix=cm).plot().figure_.savefig("artifacts/cm.png")
        mlflow.log_artifact("artifacts/cm.png")
        
        mlflow.sklearn.log_model(model, "model")
        logger.info(f"🚀 SUCCESS! Accuracy: {acc}")

if __name__ == "__main__":
    run()
