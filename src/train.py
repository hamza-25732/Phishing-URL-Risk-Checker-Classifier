"""
Train and evaluate Random Forest and XGBoost classifiers for
phishing URL detection, using the lexical/structural features
built in features.py.

Outputs:
  - models/random_forest.pkl
  - models/xgboost.pkl
  - models/feature_columns.json
  - reports/model_comparison.json
  - reports/confusion_matrices.png
  - reports/feature_importance.png
"""

import json
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DATA_PATH = "data/features_full.csv"
MODELS_DIR = "models"
REPORTS_DIR = "reports"

RANDOM_STATE = 42


EXCLUDED_FEATURES = {
    "url_length", "path_length", "domain_length", "num_slashes",
    "num_letters", "num_digits", "num_dots", "num_special_chars", "num_params",
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    feature_cols = [
        c for c in df.columns
        if c not in ("label", "url") and c not in EXCLUDED_FEATURES
    ]
    X = df[feature_cols]
    y = df["label"]
    return X, y, feature_cols


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    metrics["false_positive_rate"] = fp / (fp + tn)
    metrics["false_negative_rate"] = fn / (fn + tp)

    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        print(f"{k:22s}: {v:.4f}")
    print(classification_report(y_test, preds, target_names=["legit", "phishing"]))

    return metrics, cm, preds, probs


def main():
    print("Loading feature data...")
    X, y, feature_cols = load_data()
    print(f"Features: {len(feature_cols)} | Rows: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    results = {}

    # --- Random Forest ---
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=15,
        min_samples_split=30,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0
    rf_metrics, rf_cm, rf_preds, rf_probs = evaluate("Random Forest", rf, X_test, y_test)
    rf_metrics["train_time_sec"] = rf_time
    results["random_forest"] = rf_metrics

    # --- XGBoost ---
    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=2.0,
        reg_alpha=0.5,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    xgb_time = time.time() - t0
    xgb_metrics, xgb_cm, xgb_preds, xgb_probs = evaluate("XGBoost", xgb, X_test, y_test)
    xgb_metrics["train_time_sec"] = xgb_time
    results["xgboost"] = xgb_metrics

    # --- Save models ---
    with open(f"{MODELS_DIR}/random_forest.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open(f"{MODELS_DIR}/xgboost.pkl", "wb") as f:
        pickle.dump(xgb, f)
    with open(f"{MODELS_DIR}/feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open(f"{REPORTS_DIR}/model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- Confusion matrices plot ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, cm, name in zip(axes, [rf_cm, xgb_cm], ["Random Forest", "XGBoost"]):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["legit", "phishing"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["legit", "phishing"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/confusion_matrices.png", dpi=140)
    plt.close()

    # --- Feature importance plot (both models) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, model, name in zip(axes, [rf, xgb], ["Random Forest", "XGBoost"]):
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        importances = importances.sort_values(ascending=True).tail(15)
        ax.barh(importances.index, importances.values, color="#4C72B0")
        ax.set_title(f"{name} — Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/feature_importance.png", dpi=140)
    plt.close()

    print("\nSaved models to models/, reports to reports/")
    print("\n=== Summary ===")
    print(pd.DataFrame(results).T.round(4))


if __name__ == "__main__":
    main()
