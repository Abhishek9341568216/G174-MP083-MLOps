import argparse

import pandas as pd

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="Inspect project dataset quality.")
    parser.add_argument("--config", default="params.yaml")
    config = load_config(parser.parse_args().config)

    df = pd.read_csv(config["data"]["train_path"])
    later = pd.read_csv(config["data"]["later_path"])

    print("\n========== 1. DATA LEAKAGE ==========")
    print("Columns absent from later:")
    print(set(df.columns) - set(later.columns))

    print("\n========== 2. UNIT MIXUP ==========")

    numeric_columns = [
        "age",
        "screen_hours",
        "apps_installed",
        "plan_gb_limit",
        "video_minutes",
        "wifi_hours",
    ]

    for column in numeric_columns:
        print(f"\n{column}")
        print(
            df.groupby("circle")[column]
            .agg(["min", "mean", "median", "max"])
            .to_string()
        )

    print("\n========== 3. SENTINEL VALUES ==========")

    for column in numeric_columns:
        print(f"\n{column}")
        for value in [-999, -1, 999, 9999]:
            count = (df[column] == value).sum()

            if count > 0:
                print(f"Suspicious value: {value}, Count: {count}")

    print("\n========== 4. DUPLICATE ROWS ==========")
    print("Exact duplicate rows:", df.duplicated().sum())

    print("\n========== 5. STUCK SENSOR ==========")

    df["date"] = pd.to_datetime(df["date"])

    for column in numeric_columns:
        daily = df.groupby("date")[column].agg(
            ["min", "max", "count"]
        )

        stuck_days = daily[
            (daily["min"] == daily["max"]) &
            (daily["count"] > 1)
        ]

        print(f"\n{column}:")
        print(stuck_days.head(20).to_string())

    print("\n========== 6. TARGET TYPOS ==========")

    print(df["gb_per_day"].describe())

    print("\nLargest target values:")
    print(
        df.nlargest(
            20, "gb_per_day"
        )[["record_id", "date", "gb_per_day"]].to_string(index=False)
    )

    print("\n========== 7. SPELLING VARIANTS ==========")

    for column in ["phone_tier", "circle"]:
        print(f"\n{column}:")
        print(df[column].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
