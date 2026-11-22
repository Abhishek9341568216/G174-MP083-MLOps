"""Fit and register a candidate, with held-out and optional promotion evaluation."""
import argparse
import logging
import os
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .cleaning import NormalizeFeatures, prepare_labeled
from .config import COLUMNS, FEATURES, TARGET, load_config
from .registry import approve, load_version, register, write_json

LOGGER = logging.getLogger(__name__)


def build_pipeline(config, estimator=None):
    numeric = SimpleImputer(strategy="median", keep_empty_features=True)
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    if estimator is None:
        estimator = RandomForestRegressor(
            n_estimators=config["model"]["n_estimators"],
            random_state=config["model"]["random_state"],
            n_jobs=1,
        )
    return Pipeline([
        ("normalize", NormalizeFeatures()),
        ("preprocessor", ColumnTransformer([
            ("numeric", numeric, FEATURES["numeric"]),
            ("categorical", categorical, FEATURES["categorical"]),
        ])),
        ("model", estimator),
    ])


def metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(root_mean_squared_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 and actual.nunique() > 1 else None,
    }


def feature_keys(frame):
    return pd.util.hash_pandas_object(frame[COLUMNS], index=False).map(lambda value: f"features:{value:016x}")


def training_keys(frame):
    keys = set(feature_keys(frame))
    if "record_id" in frame:
        keys.update("id:" + sha256(str(value).encode()).hexdigest() for value in frame["record_id"].dropna())
    return keys


def read_labeled(path):
    raw = pd.read_csv(path, dtype={"record_id": "string"})
    frame = prepare_labeled(raw)
    if "record_id" in frame and frame["record_id"].dropna().duplicated().any():
        raise ValueError("Conflicting rows share a record_id")
    if len(frame) < 4 or feature_keys(frame).nunique() < 2:
        raise ValueError("Insufficient valid labeled rows or distinct feature profiles")
    LOGGER.info("Retained %d of %d labeled rows from %s", len(frame), len(raw), path)
    return frame


def group_metrics(frame, predictions, minimum):
    scored = frame.assign(prediction=predictions)
    return {
        column: {
            str(group): {"rows": len(rows), **metrics(rows[TARGET], rows["prediction"])}
            for group, rows in scored.groupby(scored[column].fillna("(missing)"))
            if len(rows) >= minimum
        }
        for column in FEATURES["categorical"]
    }


def log_mlflow(config, pipeline, metadata, registry_name=None):
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow
    import mlflow.sklearn

    uri = os.environ.get("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="random_forest_candidate") as run:
        mlflow.log_params(config["model"])
        for model, scores in metadata["metrics"].items():
            mlflow.log_metrics({f"{model}_{key}": value for key, value in scores.items() if value is not None})
        input_example = pd.DataFrame([{
            "age": 25.0, "screen_hours": 6.0, "apps_installed": 42,
            "plan_gb_limit": 3.0, "video_minutes": 75.0, "wifi_hours": 4.0,
            "phone_tier": "mid", "circle": "north",
        }])
        mlflow.sklearn.log_model(
            pipeline, name="pipeline", input_example=input_example, registered_model_name=registry_name,
        )
        return run.info.run_id


def train(config, models, *, evaluation_path=None, promote_candidate=False, tracking=False, registry_name=None):
    if promote_candidate and evaluation_path is None:
        raise ValueError("Promotion requires independent evaluation data")
    data = read_labeled(config["data"]["train_path"])
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=config["model"]["test_size"], random_state=config["model"]["random_state"]
    )
    train_rows, test_rows = next(splitter.split(data, groups=feature_keys(data)))
    training, test = data.iloc[train_rows], data.iloc[test_rows]
    if min(len(training), len(test)) < 2:
        raise ValueError("Both splits need at least two rows")
    pipeline = build_pipeline(config).fit(training[COLUMNS], training[TARGET])
    baseline = build_pipeline(config, DummyRegressor(strategy="median")).fit(training[COLUMNS], training[TARGET])
    predicted = pipeline.predict(test[COLUMNS])
    policy = config["evaluation"]
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parameters": config["model"],
        "sklearn_version": sklearn.__version__,
        "target": TARGET,
        "features": COLUMNS,
        "training_rows": len(training),
        "test_rows": len(test),
        "metrics": {
            "random_forest": metrics(test[TARGET], predicted),
            "median_baseline": metrics(test[TARGET], baseline.predict(test[COLUMNS])),
        },
        "groups": group_metrics(test, predicted, policy["min_group_size"]),
        "training_keys": sorted(training_keys(training)),
    }
    if evaluation_path is not None:
        current, current_metadata = load_version(models)
        evaluation = read_labeled(evaluation_path)
        overlap = training_keys(evaluation) & (training_keys(data) | set(current_metadata["training_keys"]))
        if overlap or len(evaluation) < policy["min_retrain_rows"]:
            raise ValueError("Promotion evaluation needs sufficient independent labeled rows")
        candidate_scores = metrics(evaluation[TARGET], pipeline.predict(evaluation[COLUMNS]))
        current_scores = metrics(evaluation[TARGET], current.predict(evaluation[COLUMNS]))
        accepted = (
            candidate_scores["mae"] <= current_scores["mae"] - policy["min_mae_improvement"]
            and candidate_scores["rmse"] <= current_scores["rmse"] + policy["max_rmse_increase"]
        )
        metadata["promotion_evaluation"] = {
            "current_version": current_metadata["version"],
            "rows": len(evaluation),
            "candidate": candidate_scores,
            "current": current_scores,
            "accepted": accepted,
        }
    if tracking or registry_name:
        metadata["mlflow_run_id"] = log_mlflow(config, pipeline, metadata, registry_name)
    version = register(models, pipeline, metadata)
    if promote_candidate and metadata["promotion_evaluation"]["accepted"]:
        approve(models, version, expected=current_metadata["version"])
    LOGGER.info("Registered %s with held-out metrics %s", version, metadata["metrics"])
    return version, {key: value for key, value in metadata.items() if key != "training_keys"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--models", default="models")
    parser.add_argument("--report", type=Path, default=Path("reports/training.json"))
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--registry-name")
    parser.add_argument("--evaluation-data")
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    if args.promote and not args.evaluation_data:
        parser.error("--promote requires --evaluation-data")
    logging.basicConfig(level=logging.INFO)
    version, metadata = train(
        load_config(args.config), args.models, evaluation_path=args.evaluation_data,
        promote_candidate=args.promote, tracking=args.mlflow, registry_name=args.registry_name,
    )
    write_json(args.report, {**metadata, "version": version})


if __name__ == "__main__":
    main()
