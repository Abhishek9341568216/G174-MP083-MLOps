"""Normalize input without fitting any statistics."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import COLUMNS, FEATURES, TARGET, load_config


def normalize_features(frame):
    result = frame.loc[:, COLUMNS].copy()
    for column in FEATURES["numeric"]:
        values = pd.to_numeric(result[column], errors="coerce").astype(float)
        result[column] = values.where(np.isfinite(values) & (values >= 0))
    tier = result["phone_tier"].astype("string").str.strip().str.lower().str.replace(".", "", regex=False)
    result["phone_tier"] = tier.replace({"bas": "basic", "fla": "flagship"})
    result["circle"] = result["circle"].astype("string").str.strip().str.lower()
    for column in FEATURES["categorical"]:
        result[column] = result[column].replace("", pd.NA).astype(object).fillna(np.nan)
    return result


class NormalizeFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return normalize_features(X)


def prepare_labeled(frame):
    normalized = normalize_features(frame)
    normalized[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    if "record_id" in frame:
        normalized["record_id"] = frame["record_id"]
    normalized = normalized.loc[np.isfinite(normalized[TARGET]) & normalized[TARGET].ge(0)]
    return normalized.drop_duplicates().reset_index(drop=True)


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--output", type=Path, default=Path("data/processed/train_clean.csv"))
    args = parser.parse_args()
    raw = pd.read_csv(load_config(args.config)["data"]["train_path"], dtype={"record_id": "string"})
    cleaned = prepare_labeled(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output, index=False)
    print(f"Retained {len(cleaned)} of {len(raw)} rows. Saved {args.output}")


if __name__ == "__main__":
    main()
