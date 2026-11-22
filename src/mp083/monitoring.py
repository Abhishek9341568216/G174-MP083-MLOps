"""Compare two feature samples and flag configured distribution changes."""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

from .cleaning import normalize_features
from .config import COLUMNS, FEATURES, load_config
from .registry import write_json


def drift_report(reference, later, thresholds):
    reference, later = normalize_features(reference), normalize_features(later)
    if reference.empty or later.empty:
        raise ValueError("Both drift samples must contain rows")
    report = {}
    for column in COLUMNS:
        a, b = reference[column], later[column]
        missing_delta = float(b.isna().mean() - a.isna().mean())
        a, b = a.dropna(), b.dropna()
        statistic = None
        if column in FEATURES["numeric"]:
            method = "numeric_ks"
            if len(a) and len(b):
                statistic = float(ks_2samp(a, b, method="asymp").statistic)
        else:
            method = "categorical_tv"
            if len(a) and len(b):
                differences = a.value_counts(normalize=True).sub(b.value_counts(normalize=True), fill_value=0)
                statistic = float(differences.abs().sum() / 2)
        report[column] = {
            "method": method, "statistic": statistic, "threshold": thresholds[method],
            "nonmissing_reference": len(a), "nonmissing_later": len(b),
            "missing_delta": missing_delta, "missing_threshold": thresholds["missing_delta"],
            "flagged": abs(missing_delta) >= thresholds["missing_delta"]
            or (statistic is not None and statistic >= thresholds[method]),
        }
    return {
        "reference_rows": len(reference),
        "later_rows": len(later),
        "flagged_features": [column for column, result in report.items() if result["flagged"]],
        "features": report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--reference", help="Reference CSV. Defaults to the configured raw training pool.")
    parser.add_argument("--later")
    parser.add_argument("--output", type=Path, default=Path("reports/drift.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    report = drift_report(
        pd.read_csv(args.reference or config["data"]["train_path"]),
        pd.read_csv(args.later or config["data"]["later_path"]), config["monitoring"],
    )
    write_json(args.output, report)


if __name__ == "__main__":
    main()
