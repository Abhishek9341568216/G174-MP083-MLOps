from pathlib import Path

import yaml

FEATURES = {
    "numeric": ["age", "screen_hours", "apps_installed", "plan_gb_limit", "video_minutes", "wifi_hours"],
    "categorical": ["phone_tier", "circle"],
}
COLUMNS = FEATURES["numeric"] + FEATURES["categorical"]
TARGET = "gb_per_day"


def load_config(path="params.yaml"):
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if config["features"] != FEATURES or config["target"] != TARGET:
        raise ValueError("Training features and target must match the API contract")
    model = config["model"]
    if not 0 < model["test_size"] < 1 or model["n_estimators"] < 1:
        raise ValueError("Invalid test size or estimator count")
    return config
