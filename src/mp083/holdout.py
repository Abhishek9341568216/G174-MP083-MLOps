"""Score holdout rows in input order and report invalid row numbers."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError

from app.schemas import PredictionInput

from .config import COLUMNS, TARGET, load_config
from .registry import load_version, write_json


def score(frame, pipeline):
    features = frame[COLUMNS]
    valid, positions, rejected = [], [], []
    for position, values in enumerate(features.to_dict(orient="records")):
        try:
            prediction_input = PredictionInput.model_validate(values)
        except ValidationError as error:
            rejected.append({
                "data_row": position + 1,
                "fields": [item["loc"][0] for item in error.errors(include_input=False)],
            })
        else:
            valid.append(prediction_input.model_dump())
            positions.append(position)

    output = frame.iloc[positions][["record_id"]].copy() if "record_id" in frame else pd.DataFrame(index=positions)
    predictions = pipeline.predict(pd.DataFrame(valid, columns=COLUMNS)) if valid else np.array([], dtype=float)
    if predictions.shape != (len(valid),) or not np.isfinite(predictions).all():
        raise ValueError("Model returned invalid holdout predictions")
    output[TARGET] = predictions
    return output, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="params.yaml")
    parser.add_argument("--models", default="models")
    parser.add_argument("--version")
    parser.add_argument("--output", type=Path, default=Path("predictions/holdout_predictions.csv"))
    parser.add_argument("--rejections", type=Path, default=Path("reports/holdout_rejections.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    pipeline, metadata = load_version(args.models, args.version)
    frame = pd.read_csv(config["data"]["holdout_path"], dtype={"record_id": "string"})
    predictions, rejected = score(frame, pipeline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    write_json(args.rejections, {"version": metadata["version"], "scored": len(predictions), "rejected": rejected})


if __name__ == "__main__":
    main()
