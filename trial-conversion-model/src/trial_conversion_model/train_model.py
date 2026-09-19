"""
Training the XGBoost trial-conversion model and saving it for hand-off.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from trial_conversion_model.data_engineering import PROCESSED_DATA_DIR

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

FEATURES = [
    "sessions_3d",
    "active_days_3d",
    "day1_share",
    "listen_share",
    "avg_session_minutes",
    "total_minutes_3d",
    "country",
    "device_type",
]
FLAG_THRESHOLD = 0.35


def train_model(df: pd.DataFrame) -> tuple[XGBClassifier, dict]:
    features = pd.get_dummies(df[FEATURES], columns=["country", "device_type"])
    target = df["converted"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, stratify=target, random_state=42
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=8,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="auc",
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    flagged = probs < FLAG_THRESHOLD

    auc = roc_auc_score(y_test, probs)
    print("XGBoost AUC:", round(auc, 4))

    metrics = {
        "auc": auc,
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, digits=3, output_dict=True),
        "flagged_for_intervention": int(flagged.sum()),
        "total_test_trials": int(len(probs)),
        "conversion_rate_flagged": float(y_test[flagged].mean()),
        "conversion_rate_not_flagged": float(y_test[~flagged].mean()),
        "feature_importances": dict(
            zip(features.columns, model.feature_importances_.tolist())
        ),
    }

    return model, metrics


def train_and_save(
    input_path: Path = PROCESSED_DATA_DIR / "trials_processed.csv",
    output_dir: Path = MODELS_DIR,
) -> tuple[XGBClassifier, dict]:
    df = pd.read_csv(input_path)
    model, metrics = train_model(df)

    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(output_dir / "model.json")

    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return model, metrics


if __name__ == "__main__":
    train_and_save()
