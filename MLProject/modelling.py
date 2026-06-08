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

# Gunakan Agg agar tidak error di server tanpa monitor
matplotlib.use("Agg") 
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

def run():
    # 1. AMBIL ARGUMEN (Wajib sama dengan MLProject file)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    # 2. LOADING DATA
    # Pastikan file ini ada di dalam folder MLProject
    df = pd.read_csv("telco_churn_preprocessed.csv")
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # 3. TRAINING & LOGGING
    # Kita tidak pakai 'with mlflow.start_run()' karena sudah dibuat oleh 'mlflow run'
    model = RandomForestClassifier(
        n_estimators=args.n_estimators, 
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=args.random_state
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    
    # Log params & metrics (Akan otomatis masuk ke run yang sedang aktif)
    mlflow.log_params(vars(args))
    mlflow.log_metric("accuracy", acc)
    
    # Simpan Artefak ke folder lokal 'artifacts'
    os.makedirs("artifacts", exist_ok=True)
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
    fig.savefig("artifacts/cm.png")
    
    # Upload artefak ke MLflow
    mlflow.log_artifacts("artifacts")
    
    # Simpan Model
    mlflow.sklearn.log_model(model, "model")
    logger.info(f"✅ SUCCESS! Accuracy: {acc}")

if __name__ == "__main__":
    run()
