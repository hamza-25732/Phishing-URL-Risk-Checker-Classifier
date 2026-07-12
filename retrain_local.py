"""
Standalone retrain script — run this locally to regenerate xgboost.pkl and
random_forest.pkl from scratch using features_full.csv. This avoids any
issue with .pkl files getting corrupted during download/transfer.

Usage (from the phishing_url_classifier root folder):
    pip install -r requirements.txt
    python retrain_local.py

Takes under a minute on a normal laptop.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "features_full.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

EXCLUDED_FEATURES = {
    "url_length", "path_length", "domain_length", "num_slashes",
    "num_letters", "num_digits", "num_dots", "num_special_chars", "num_params",
}

RANDOM_STATE = 42


def main():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    feature_cols = [
        c for c in df.columns
        if c not in ("label", "url") and c not in EXCLUDED_FEATURES
    ]
    X = df[feature_cols]
    y = df["label"]
    print(f"Rows: {len(df)} | Features: {len(feature_cols)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=15,
        min_samples_split=30, n_jobs=-1, random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    print("  RF test accuracy:", rf.score(X_test, y_test))

    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_lambda=2.0, reg_alpha=0.5, eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    print("  XGB test accuracy:", xgb.score(X_test, y_test))

    with open(MODELS_DIR / "random_forest.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open(MODELS_DIR / "xgboost.pkl", "wb") as f:
        pickle.dump(xgb, f)
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"\nSaved models to {MODELS_DIR}/")
    print("You can now run: cd app && streamlit run app.py")


if __name__ == "__main__":
    main()
