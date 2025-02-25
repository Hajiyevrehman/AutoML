# pipeline/visualizations.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_curve, auc

def ensure_visualizations_dir():
    """
    Ensure that a folder named 'visualizations' exists at the root level
    for saving plots.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    vis_dir = os.path.join(base_dir, "visualizations")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    return vis_dir


def visualize_pre_training(df, target_col='Survived', sample_n=2000):
    """
    Saves:
      1. Target distribution plot
      2. Correlation heatmap
      3. Pairplot (sampled)
      4. Distribution of numeric features
      5. Boxplot of numeric features
    """
    vis_dir = ensure_visualizations_dir()
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # 1) Target distribution
    if target_col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=df[target_col])
        plt.title(f"Distribution of {target_col}")
        plt.savefig(os.path.join(vis_dir, "target_distribution.png"))
        plt.close()

    # 2) Correlation heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=False, cmap="YlGnBu")
        plt.title("Correlation Heatmap")
        plt.savefig(os.path.join(vis_dir, "correlation_heatmap.png"))
        plt.close()

    # 3) Pairplot (sampled if large)
    sample_df = df.sample(min(sample_n, len(df)), random_state=42)
    if len(numeric_cols) > 1:
        try:
            pairplot = sns.pairplot(sample_df, vars=numeric_cols,
                                    hue=target_col if target_col in df.columns else None)
            pairplot.fig.suptitle("Pairplot of Numeric Features (sampled)", y=1.02)
            pairplot.savefig(os.path.join(vis_dir, "pairplot.png"))
            plt.close()
        except Exception as e:
            print("Pairplot visualization failed:", e)

    # 4) Distribution of each numeric feature
    for col in numeric_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(data=df, x=col, hue=target_col if target_col in df.columns else None, kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(os.path.join(vis_dir, f"dist_{col}.png"))
        plt.close()

    # 5) Boxplot of each numeric feature
    for col in numeric_cols:
        plt.figure(figsize=(6,4))
        sns.boxplot(data=df, x=target_col if target_col in df.columns else None, y=col)
        plt.title(f"Boxplot of {col} by {target_col}")
        plt.savefig(os.path.join(vis_dir, f"boxplot_{col}.png"))
        plt.close()


def visualize_post_training(model, X_test, y_test, target_col='Survived'):
    """
    Post-training plots:
      1. Confusion Matrix
      2. Classification Report (printed to console, also saved as .txt)
      3. (If binary classification) ROC curve
    """
    vis_dir = ensure_visualizations_dir()

    # 1) Confusion Matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(vis_dir, "confusion_matrix.png"))
    plt.close()

    # 2) Classification Report
    clf_report = classification_report(y_test, y_pred)
    print("\n=== Classification Report ===")
    print(clf_report)
    # save to .txt
    with open(os.path.join(vis_dir, "classification_report.txt"), "w") as f:
        f.write(clf_report)

    # 3) ROC Curve (only if it's a binary classification)
    # Check number of unique labels in y_test
    unique_labels = np.unique(y_test)
    if len(unique_labels) == 2:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(vis_dir, "roc_curve.png"))
            plt.close()
        else:
            print("Model has no predict_proba; skipping ROC curve.")
