import numpy as np
import pandas as pd
import pytest

from src.mp083.cleaning import prepare_labeled
from src.mp083.config import COLUMNS, load_config
from src.mp083.holdout import score
from src.mp083.monitoring import drift_report
from src.mp083.registry import approve, load_version
from src.mp083.train import feature_keys, train


def test_cleaning_does_not_impute_or_include_bill(sample):
    sample.loc[0, "age"] = -1
    sample.loc[1, "gb_per_day"] = np.inf
    rows = pd.concat([sample, sample.iloc[[0]]], ignore_index=True)
    cleaned = prepare_labeled(rows)
    assert len(cleaned) == 59
    assert np.isnan(cleaned.loc[0, "age"])
    assert cleaned.loc[0, "phone_tier"] == "mid"
    assert cleaned.loc[2, "phone_tier"] == "basic"
    assert "gb_from_monthly_bill" not in cleaned


def test_config_rejects_leakage_column(tmp_path):
    config = tmp_path / "params.yaml"
    from pathlib import Path

    config.write_text(Path("params.yaml").read_text().replace("- age", "- gb_from_monthly_bill"))
    with pytest.raises(ValueError, match="API contract"):
        load_config(config)


def test_fit_uses_training_medians_and_predicts(approved, sample):
    models, version, metadata = approved
    pipeline, detail = load_version(models)
    assert detail["version"] == version
    frame = prepare_labeled(sample)
    training = frame.loc[feature_keys(frame).isin(detail["training_keys"])]
    assert len(training) == metadata["training_rows"]
    assert metadata["test_rows"] + len(training) == len(sample)
    imputer = pipeline.named_steps["preprocessor"].named_transformers_["numeric"]
    assert imputer.statistics_[0] == training.age.median()
    assert metadata["metrics"]["random_forest"]["mae"] < metadata["metrics"]["median_baseline"]["mae"]
    assert metadata["groups"]["phone_tier"]["basic"]["rows"] > 0
    predictions = pipeline.predict(sample[COLUMNS].iloc[[0, -1]])
    assert predictions[0] < predictions[1]


def test_retrain_gates_promotion_and_rolls_back(tmp_path, approved, sample, settings):
    models, initial, _ = approved
    with pytest.raises(ValueError, match="independent"):
        train(settings, models, evaluation_path=settings["data"]["train_path"])
    evaluation = sample.assign(record_id=lambda data: "new-" + data.record_id, age=lambda data: data.age + 0.1)
    path = tmp_path / "evaluation.csv"
    evaluation.to_csv(path, index=False)
    settings["evaluation"]["min_mae_improvement"] = 100
    rejected, report = train(settings, models, evaluation_path=path, promote_candidate=True)
    assert not report["promotion_evaluation"]["accepted"]
    assert load_version(models)[1]["version"] == initial
    assert load_version(models, rejected)[1]["version"] == rejected
    settings["evaluation"]["min_mae_improvement"] = 0
    promoted, report = train(settings, models, evaluation_path=path, promote_candidate=True)
    assert report["promotion_evaluation"]["accepted"]
    assert load_version(models)[1]["version"] == promoted
    with pytest.raises(ValueError, match="changed"):
        approve(models, rejected, expected=initial)
    approve(models, initial)
    assert load_version(models)[1]["version"] == initial


def test_drift_reports_numeric_categorical_and_missing(sample, settings):
    later = sample.assign(phone_tier="flagship", screen_hours=23, wifi_hours=np.nan)
    report = drift_report(sample, later, settings["monitoring"])
    assert set(report["flagged_features"]) == {"phone_tier", "screen_hours", "wifi_hours"}
    assert report["features"]["phone_tier"]["statistic"] == 1
    assert report["features"]["wifi_hours"]["statistic"] is None


def test_holdout_preserves_order_ids_and_rejections(approved, sample):
    pipeline, _ = load_version(approved[0])
    rows = sample.iloc[[2, 0, 3]].copy()
    rows.loc[0, "age"] = -1
    output, rejected = score(rows, pipeline)
    assert list(output.columns) == ["record_id", "gb_per_day"]
    assert output.record_id.tolist() == ["002", "003"]
    np.testing.assert_allclose(output.gb_per_day, pipeline.predict(rows.iloc[[0, 2]][COLUMNS]))
    assert rejected == [{"data_row": 2, "fields": ["age"]}]
