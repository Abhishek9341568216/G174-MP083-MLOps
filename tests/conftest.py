import pandas as pd
import pytest

from src.mp083.config import load_config
from src.mp083.registry import approve
from src.mp083.train import train


@pytest.fixture
def sample():
    return pd.DataFrame([
        {
            "record_id": f"{index:03}", "age": 21 + index, "screen_hours": 1 + index / 20,
            "apps_installed": 10 + index, "plan_gb_limit": 5, "video_minutes": index * 3,
            "wifi_hours": 4, "phone_tier": "Bas." if index % 2 else "Mid",
            "circle": "North" if index % 3 else "South", "gb_per_day": 1 + index / 20,
            "gb_from_monthly_bill": 30 + index * 1.5,
        }
        for index in range(60)
    ])


@pytest.fixture
def settings(tmp_path, sample):
    config = load_config()
    path = tmp_path / "train.csv"
    sample.to_csv(path, index=False)
    config["data"]["train_path"] = str(path)
    config["model"]["n_estimators"] = 6
    config["evaluation"]["min_group_size"] = 2
    return config


@pytest.fixture
def approved(tmp_path, settings):
    models = tmp_path / "models"
    version, metadata = train(settings, models)
    approve(models, version)
    return models, version, metadata
