import json
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from trial_conversion_model.data import load_processed
from trial_conversion_model.features import TARGET

MODEL_DIR = Path("models")
TEST_SIZE = 0.25
RANDOM_STATE = 100
DEFAULT_TRACKING_URI = "http://127.0.0.1:5001"
EXPERIMENT_NAME = "trial-conversion-model"

PARAMS = {
    "n_estimators": 400,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_child_weight": 8,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "eval_metric": "auc",
}


def train(
    model_dir: Path = MODEL_DIR,
    params: dict | None = None,
    run_name: str | None = None,
) -> dict:
    """Train the trial conversion model from the processed training table."""
    params = params or PARAMS

    # Every run, regardless of the attempt: where to log, and the data.
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(EXPERIMENT_NAME)

    table = load_processed()
    X = table.drop(columns=[TARGET])
    y = table[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # This particular attempt: its settings, its result, its model.
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {**params, "test_size": TEST_SIZE, "random_state": RANDOM_STATE}
        )

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        mlflow.log_metrics(
            {"test_auc": auc, "n_train": len(X_train), "n_test": len(X_test)}
        )
        mlflow.xgboost.log_model(model, name="model")

    # Every run, regardless of the attempt: local artifacts the API loads.
    model_dir.mkdir(exist_ok=True)
    model.save_model(model_dir / "model.json")
    metrics = {
        "test_auc": round(float(auc), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": list(X.columns),
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
