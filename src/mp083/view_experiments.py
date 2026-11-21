import mlflow

mlflow.set_tracking_uri("./mlruns")

experiment = mlflow.get_experiment_by_name(
    "G174_MP083_Mobile_Data"
)

if experiment is None:
    print("Experiment not found")
else:
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    columns = [
        "run_id",
        "metrics.mae",
        "metrics.rmse",
        "metrics.r2"
    ]

    print(runs[columns].to_string(index=False))