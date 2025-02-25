# pipeline/hyper_tuning.py

import optuna
import numpy as np
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from optuna.exceptions import TrialPruned

warnings.filterwarnings("ignore", category=UserWarning)

def tune_model(model_name: str, X, y, n_trials=20, n_splits=5, random_state=42):
    """
    Tune hyperparameters for a given model_name using K-fold cross-validation (classification).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier,
        ExtraTreesClassifier, AdaBoostClassifier
    )
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
    from sklearn.linear_model import RidgeClassifier

    # Try to import XGBoost
    try:
        from xgboost import XGBClassifier
        XGB_AVAILABLE = True
    except ImportError:
        XGBClassifier = None
        XGB_AVAILABLE = False

    def create_model(trial):
        if model_name == "Logistic Regression":
            C = trial.suggest_float("C", 1e-3, 10, log=True)
            solver = trial.suggest_categorical("solver", ["liblinear", "lbfgs"])
            return LogisticRegression(C=C, solver=solver, max_iter=1000)
        elif model_name == "Decision Tree":
            max_depth = trial.suggest_int("max_depth", 2, 20)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            return DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
        elif model_name == "Gaussian NB":
            return GaussianNB()
        elif model_name == "Random Forest":
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 2, 20)
            return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        elif model_name == "Gradient Boosting":
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 2, 20)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            return GradientBoostingClassifier(
                n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate
            )
        elif model_name == "Extra Trees":
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 2, 20)
            return ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        elif model_name == "AdaBoost":
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 1.0, log=True)
            return AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
        elif model_name == "SVM":
            C = trial.suggest_float("C", 1e-3, 10, log=True)
            gamma = trial.suggest_float("gamma", 1e-4, 1e-1, log=True)
            return SVC(C=C, gamma=gamma, probability=True)
        elif model_name == "KNN":
            n_neighbors = trial.suggest_int("n_neighbors", 3, 15)
            return KNeighborsClassifier(n_neighbors=n_neighbors)
        elif model_name == "QDA":
            return QuadraticDiscriminantAnalysis()
        elif model_name == "LDA":
            return LinearDiscriminantAnalysis()
        elif model_name == "Ridge Classifier":
            alpha = trial.suggest_float("alpha", 0.1, 10, log=True)
            return RidgeClassifier(alpha=alpha)
        elif model_name == "XGBoost":
            if not XGB_AVAILABLE:
                raise ValueError("XGBoost not installed.")
            n_estimators = trial.suggest_int("n_estimators", 50, 300)
            max_depth = trial.suggest_int("max_depth", 2, 20)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            return XGBClassifier(
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate, use_label_encoder=False,
                eval_metric="logloss", random_state=random_state
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", pruner=pruner)

    def objective(trial):
        model = create_model(trial)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        f1_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            f1 = f1_score(y_val, preds, average='macro')
            f1_scores.append(f1)

            trial.report(f1, fold_idx)
            if trial.should_prune():
                raise TrialPruned()

        return np.mean(f1_scores)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = study.best_params
    best_score = study.best_value

    print(f"\n[Optuna] Best params for {model_name}: {best_params}")
    print(f"[Optuna] Best F1 = {best_score:.4f}")
    return best_params, best_score


def tune_candidates(model_names: list, X, y, n_trials=20, n_splits=5):
    """
    Loop over each model name, tune hyperparams, return results.
    """
    results = {}
    for name in model_names:
        print(f"\n===== Tuning {name} =====")
        try:
            best_params, best_score = tune_model(
                name, X, y, n_trials=n_trials, n_splits=n_splits
            )
            results[name] = (best_params, best_score)
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")
    return results


