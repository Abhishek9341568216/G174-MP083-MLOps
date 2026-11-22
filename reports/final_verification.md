# Final verification

Verified on Linux with Python 3.12 on branch `feature/complete-mlops-pipeline`.

| Check | Status | Result |
| --- | --- | --- |
| pytest | Passed | 8 tests passed. Starlette TestClient emitted one AnyIO deprecation warning. |
| Ruff and dependency consistency | Passed | Ruff and pip check succeeded. |
| Python imports and configuration | Passed | Application and pipeline modules imported, byte-compiled and loaded params.yaml. |
| Cleaning, training and model loading | Passed with fixtures | Training-only imputation, baseline comparison, version selection, promotion rejection and acceptance, and rollback exercised. |
| MLflow | Passed with fixtures | File-backed experiment logging worked. SQLite-backed MLflow registered a fixture pipeline as version 1 after pinning SQLAlchemy to a compatible release. |
| API startup, readiness, prediction and logs | Passed with fixtures | TestClient covered serving and missing-model paths. Docker served real outputs from the approved fixture model and emitted JSON application events. |
| Holdout CLI and drift CLI | Passed with fixtures | Four ordered fixture IDs were preserved in the batch CSV. A fixture drift report was produced. |
| Docker | Passed locally | Built with the environment's trusted CA bundle, then verified model readiness and prediction as the image's non-root user. |
| Kubernetes | Schema passed | Deployment, Service and ConfigMap passed Kubernetes v1.33 strict JSON schema validation. Cluster deployment was not run. |
| DVC status and pull | Blocked | DVC status found all four outputs absent from cache. DVC pull failed because no remote or local cached data was available. |
| Project-data checks | Blocked | No raw or processed CSVs, historical model or historical MLflow store were available. Project training, outlier and overlap investigation, fairness results, drift results and holdout predictions remain unverified. |
| Windows and Codespaces | Not run | Linux tracking and SQLite registry were tested locally. |
| GitHub Actions and remote push | Not run | GitHub rejected the configured token and the dry-run push reported no usable credential. |

The historical baseline supplied with the project was MAE 0.359, RMSE 0.4421 and R² 0.8618 with 800 training and 200 test rows, run `6be0748e1e9548eaa67dd30ba474582c`. The run and its metrics could not be independently verified. Fixture results establish that the pipeline runs. They provide no project performance estimate.

## Git changes

- `db5f757` implements the pipeline, API, tests and deployment configuration.
- A follow-up documentation commit adds the README and both submission reports.
- The staged changes were checked for credential patterns, dataset and model files, experiment stores, virtual environments, temporary files and binary additions.
