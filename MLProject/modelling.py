import argparse
import logging
import os
import sys
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") 

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# --- KONSTANTA ---
DATA_PATH = os.path.join(os.path.dirname(__file__), "telco_churn_preprocessed.csv")
TARGET_COLUMN = "Churn"
EXPERIMENT_NAME = "Telco_Churn_CI_FirahMaulida"

def run(args):
    # --- 🛡️ KUNCI ANTI LOGIN (UNTUK GITHUB ACTIONS) 🛡️ ---
    token = os.getenv("DAGSHUB_TOKEN")
    repo_owner = "firahmaulida"
    repo_name = "Eksperimen_SML_FirahMaulida"
    
    if token:
        # Jika di GitHub, kita set manual tanpa manggil dagshub.init yang rewel
        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"✅ CI Mode: MLflow Tracking URI set to {tracking_uri}")
    else:
        # Jika di laptop sendiri, pakai cara biasa
        import dagshub
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    
    mlflow.set_experiment(EXPERIMENT_NAME)

    # --- LOADING DATA ---
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- TRAINING ---
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_metric("accuracy", acc)
        
        # Save Artifact
        os.makedirs("artifacts", exist_ok=True)
        cm = confusion_matrix(y_test, model.predict(X_test))
        ConfusionMatrixDisplay(confusion_matrix=cm).plot().figure_.savefig("artifacts/cm.png")
        mlflow.log_artifact("artifacts/cm.png")
        
        mlflow.sklearn.log_model(model, "model")
        logger.info(f"🚀 Success! Accuracy: {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()
    run(args)
