# pipeline/model_selector.py

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGB_AVAILABLE = False

def select_models(data_size: int) -> list:
    """
    Rule-based selection of three candidate models based on dataset size.
    """
    all_models = [
        "Logistic Regression", "Decision Tree", "Gaussian NB",
        "Random Forest", "Gradient Boosting", "Extra Trees",
        "AdaBoost", "SVM", "KNN", "QDA", "LDA", "Ridge Classifier"
    ]
    if XGB_AVAILABLE:
        all_models.append("XGBoost")

    if data_size < 5000:
        candidates = ["Logistic Regression", "Decision Tree", "Gaussian NB"]
    elif data_size < 50000:
        candidates = ["Random Forest", "Gradient Boosting", "SVM"]
    else:
        if XGB_AVAILABLE:
            candidates = ["Random Forest", "Extra Trees", "XGBoost"]
        else:
            candidates = ["Random Forest", "Extra Trees", "Gradient Boosting"]
    return candidates

def get_model_instance(model_name: str, params: dict = None):
    """
    Instantiate a model with optional hyperparameters for classification.
    """
    params = params or {}
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, **params)
    elif model_name == "Decision Tree":
        return DecisionTreeClassifier(**params)
    elif model_name == "Gaussian NB":
        return GaussianNB(**params)
    elif model_name == "Random Forest":
        return RandomForestClassifier(**params)
    elif model_name == "Gradient Boosting":
        return GradientBoostingClassifier(**params)
    elif model_name == "Extra Trees":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(**params)
    elif model_name == "AdaBoost":
        from sklearn.ensemble import AdaBoostClassifier
        return AdaBoostClassifier(**params)
    elif model_name == "SVM":
        return SVC(probability=True, **params)
    elif model_name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(**params)
    elif model_name == "QDA":
        return QuadraticDiscriminantAnalysis(**params)
    elif model_name == "LDA":
        return LinearDiscriminantAnalysis(**params)
    elif model_name == "Ridge Classifier":
        from sklearn.linear_model import RidgeClassifier
        return RidgeClassifier(**params)
    elif model_name == "XGBoost":
        if XGB_AVAILABLE:
            return XGBClassifier(use_label_encoder=False, eval_metric="logloss", **params)
        else:
            raise ImportError("XGBoost not installed.")
    else:
        raise ValueError(f"Unknown model name: {model_name}")

