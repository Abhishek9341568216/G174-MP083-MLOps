import argparse

import pandas as pd

from .cleaning import normalize_features
from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="Inspect project dataset quality.")
    parser.add_argument("--config", default="params.yaml")
    config = load_config(parser.parse_args().config)

    # Load datasets
    train = pd.read_csv(config["data"]["train_path"], dtype={"record_id": "string"})
    later = pd.read_csv(config["data"]["later_path"], dtype={"record_id": "string"})
    holdout = pd.read_csv(config["data"]["holdout_path"], dtype={"record_id": "string"})

    # Dataset sizes
    print("\n========== DATASET SHAPES ==========")
    print("Train:", train.shape)
    print("Later:", later.shape)
    print("Holdout:", holdout.shape)

    # Column names
    print("\n========== COLUMNS ==========")
    print(train.columns.tolist())

    # Missing values
    print("\n========== MISSING VALUES ==========")
    print(train.isnull().sum())

    # Duplicate rows
    print("\n========== DUPLICATE ROWS ==========")
    print(train.duplicated().sum())

    # Numerical statistics
    print("\n========== NUMERICAL SUMMARY ==========")
    print(train.describe().to_string())

    # Categorical values
    print("\n========== PHONE TIER VALUES ==========")
    print(train["phone_tier"].value_counts(dropna=False))

    print("\n========== CIRCLE VALUES ==========")
    print(train["circle"].value_counts(dropna=False))

    # Compare train and holdout columns
    print("\n========== COLUMNS NOT IN HOLDOUT ==========")
    print(set(train.columns) - set(holdout.columns))

    print("\n========== CROSS-DATASET OVERLAP ==========")
    train_profiles = set(pd.util.hash_pandas_object(normalize_features(train), index=False))
    for name, frame in (("later", later), ("holdout", holdout)):
        profiles = set(pd.util.hash_pandas_object(normalize_features(frame), index=False))
        print(f"Train/{name} identical normalized profiles: {len(train_profiles & profiles)}")
        if "record_id" in train and "record_id" in frame:
            ids = set(train["record_id"].dropna()) & set(frame["record_id"].dropna())
            print(f"Train/{name} matching record IDs: {len(ids)}")


if __name__ == "__main__":
    main()
