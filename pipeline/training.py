# pipeline/training.py

import os
import time
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, f1_score
from pipeline.model_selector import select_models, get_model_instance

def get_absolute_path(file_path: str) -> str:
    if os.path.isabs(file_path):
        return file_path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(project_root, file_path)

def save_model(model, model_name):
    models_dir = get_absolute_path("models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    filepath = os.path.join(models_dir, model_name + ".pkl")
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved {model_name} to {filepath}")

def train_and_evaluate(target_col="Survived"):
    data_path = get_absolute_path("data/processed/engineered_data.csv")
    df = pd.read_csv(data_path)
    
    if target_col not in df.columns:
        print(f"Warning: {target_col} not found in columns. Available columns: {list(df.columns)}")
        return

    # ensure Survived is int for classification (if it's binary float)
    if df[target_col].dtype != int:
        # optional check if it can be cast
        try:
            df[target_col] = df[target_col].astype(int)
        except:
            print("Could not cast Survived to int. Possibly not a binary or discrete col.")
            return
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    data_size = len(df)
    candidate_models = select_models(data_size)

    results = {}
    
    for model_name in candidate_models:
        model = get_model_instance(model_name)
        print(f"\nTraining {model_name} ...")
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            loss = log_loss(y_test, y_proba)
        else:
            loss = None
        
        results[model_name] = {
            "accuracy": acc,
            "f1_score": f1,
            "log_loss": loss,
            "training_time": training_time
        }
        save_model(model, model_name.replace(" ", "_"))

    # sort by f1
    rankings = sorted(results.items(), key=lambda x: x[1]["f1_score"], reverse=True)
    print("\n==== Model Rankings (by F1) ====")
    for rank, (mname, metrics) in enumerate(rankings, start=1):
        print(f"{rank}. {mname}: F1={metrics['f1_score']:.4f}, Acc={metrics['accuracy']:.4f}, "
              f"Loss={metrics['log_loss']}, TrainTime={metrics['training_time']:.2f}s")


