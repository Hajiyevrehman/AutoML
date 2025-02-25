# main.py

import os
import pickle
import mlflow
import pandas as pd

from sklearn.model_selection import train_test_split
from pipeline.data_cleaning import DataCleaner
from pipeline.feature_engineering import FeatureEngineer
from pipeline.visualizations import visualize_pre_training, visualize_post_training
from pipeline.hyper_tuning import tune_model
from pipeline.model_selector import get_model_instance

def get_all_model_names():
    return [
        "Logistic Regression", "Decision Tree", "Gaussian NB",
        "Random Forest", "Gradient Boosting", "Extra Trees",
        "AdaBoost", "SVM", "KNN", "QDA", "LDA", "Ridge Classifier",
        "XGBoost"
    ]

def main():
    # 1) Data Cleaning
    cleaner = DataCleaner("data/raw/data.csv", file_type="csv")
    cleaned_df = cleaner.clean_data()
    cleaner.save_cleaned_data("cleaned_data.csv")

    # 2) Feature Engineering
    fe = FeatureEngineer("data/processed/cleaned_data.csv", file_type="csv")
    fe.run()
    fe.save_engineered_data("engineered_data.csv")

    # 3) Load final data & ensure Survived is correct
    engineered_path = os.path.join("data", "processed", "engineered_data.csv")
    df = pd.read_csv(engineered_path)
    target_col = "Survived"
    if target_col not in df.columns:
        # rename last column to Survived if needed
        maybe_target = df.columns[-1]
        print(f"'{target_col}' not found. Renaming '{maybe_target}' -> '{target_col}'")
        df.rename(columns={maybe_target: target_col}, inplace=True)

    # ensure Survived is integer-coded
    if df[target_col].dtype != int:
        try:
            df[target_col] = df[target_col].astype(int)
        except:
            print("Could not cast Survived to int. Exiting.")
            return

    # 4) Pre-training Visualizations
    visualize_pre_training(df, target_col=target_col, sample_n=2000)

    # 5) Train/Val/Test Split
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    # -> 60% train, 20% val, 20% test

    # 6) Hyperparameter Tuning
    model_names = get_all_model_names()
    results = []
    for model_name in model_names:
        print(f"\n=== Tuning {model_name} ===")
        try:
            best_params, best_f1 = tune_model(
                model_name, X_train, y_train, n_trials=10, n_splits=3
            )
        except Exception as e:
            print(f"Skipping {model_name} due to error: {e}")
            continue

        # Train model on TRAIN
        model = get_model_instance(model_name, best_params)
        model.fit(X_train, y_train)

        # Evaluate on VAL
        val_score = model.score(X_val, y_val)
        print(f"Validation Accuracy for {model_name} = {val_score:.4f}")

        results.append({
            "model_name": model_name,
            "val_score": val_score,
            "best_params": best_params
        })

    if not results:
        print("No models were successfully tuned.")
        return

    # pick best by val_score
    best_entry = max(results, key=lambda x: x["val_score"])
    best_model_name = best_entry["model_name"]
    best_params = best_entry["best_params"]
    print(f"\nBest model on validation: {best_model_name} (val_score = {best_entry['val_score']:.4f})")

    # 7) Retrain best model on (train+val), then evaluate on test
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    final_model = get_model_instance(best_model_name, best_params)
    final_model.fit(X_trainval, y_trainval)
    test_score = final_model.score(X_test, y_test)
    print(f"Final model test accuracy = {test_score:.4f}")

    # 8) Post-training Visualizations
    visualize_post_training(final_model, X_test, y_test, target_col=target_col)

    # 9) Log to MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    with mlflow.start_run():
        mlflow.log_param("best_model", best_model_name)
        for k, v in best_params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("test_accuracy", test_score)

        # Save final model
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{best_model_name.replace(' ', '_')}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(final_model, f)
        mlflow.log_artifact(model_path)

        print(f"\n[MLflow] Final model saved to: {model_path}")

if __name__ == "__main__":
    main()
