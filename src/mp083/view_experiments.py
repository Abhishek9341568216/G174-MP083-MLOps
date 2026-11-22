"""Display tracked experiment metrics."""
import os

from .config import load_config


def main():
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow

    settings = load_config()["mlflow"]
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", settings["tracking_uri"]))
    experiment = mlflow.get_experiment_by_name(settings["experiment_name"])
    if experiment is None:
        print("Experiment not found")
        return
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    columns = ["run_id", "metrics.mae", "metrics.rmse", "metrics.r2",
               "metrics.random_forest_mae", "metrics.random_forest_rmse", "metrics.random_forest_r2"]
    print(runs.reindex(columns=columns).to_string(index=False))


if __name__ == "__main__":
    main()
