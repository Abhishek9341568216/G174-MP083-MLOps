# G174-MP083 · Mobile Data Consumption Prediction

SCSE3040 MLOps mini project . The goal is to estimate mobile data consumption in GB per day from usage, subscription and device characteristics.

The prediction contract uses `age`, `screen_hours`, `apps_installed`, `plan_gb_limit`, `video_minutes`, `wifi_hours`, `phone_tier` and `circle`. The target is `gb_per_day`. The training CSV is labeled, the later CSV is used to monitor distribution changes, and the holdout CSV supplies features for prediction. The billing-derived `gb_from_monthly_bill` could encode the target and is excluded by the feature allowlist. Dates and record IDs are also excluded as predictors.

```text
DVC CSVs → schema checks and deterministic cleaning → grouped train/test split
  → training-only imputation and encoding → Random Forest + median baseline
  → held-out metrics and group analysis → candidate version → approved version
  → FastAPI / Docker / Kubernetes or batch holdout scoring
Later CSV + reference CSV → drift report
Independent labeled CSV + candidate + approved model → promotion evaluation
```

Python, pandas, scikit-learn and joblib support preprocessing and inference. DVC tracks data, MLflow tracks experiments, and FastAPI and Uvicorn serve predictions. pytest and Ruff run in GitHub Actions.

| Path | Contents |
| --- | --- |
| `src/mp083/` | Analysis, cleaning, training, versioning, monitoring and batch scoring |
| `app/` | API, validation and startup model loading |
| `data/` | DVC pointers for raw and processed data |
| `models/` | Local model versions and approval pointer, generated |
| `reports/` | Verification documents and generated JSON |
| `predictions/` | Generated holdout CSV |
| `k8s/`, `.github/workflows/`, `.devcontainer/` | Deployment, CI and Codespaces |

## Install

Run commands from the repository root with Python 3.12. Activate the virtual environment before running the remaining examples.

PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Restore and inspect data

Obtain the original CSVs or a DVC remote from the team. With a configured remote, run `dvc pull`. Alternatively place the original files at the paths in `params.yaml`. The repository currently supplies DVC metadata and checksums. Use `dvc status` to check the workspace.

```bash
python -m src.mp083.data_analysis
python -m src.mp083.detect_issues
python -m src.mp083.cleaning --output reports/train_prepared.csv
```

The standalone cleaning command defaults to `data/processed/train_clean.csv`. Use the report path in the example to preserve the existing DVC-tracked snapshot. The new training command reads raw training data to avoid medians fitted by the historical cleaner. Cleaning normalizes existing phone-tier aliases, trims and lowercases categorical values, treats negative or non-finite feature values as missing, and removes invalid labels and duplicate normalized labeled rows. Conflicting duplicate record IDs cause training to stop. The diagnostic scripts help inspect sentinel values, potential unit inconsistencies, target outliers, spelling variants and stuck values. Review their output before changing units or removing positive outliers.

## Train and track

```bash
python -m src.mp083.train
python -m src.mp083.train --mlflow
python -m src.mp083.view_experiments
```

Run the desired training command once. Both versions train and save a candidate. The second also logs to MLflow. The reproducible split groups identical normalized feature profiles so they stay on one side. Imputation and one-hot encoding fit on the training partition. The frozen pipeline handles unfamiliar categories and the report compares Random Forest with a median predictor using MAE, RMSE and R². An undefined R² is represented by null.

The previously reported baseline was MAE 0.359, RMSE 0.4421 and R² 0.8618 on 800 training and 200 test rows, run `6be0748e1e9548eaa67dd30ba474582c`. Its artifacts were unavailable during verification. The current split and preprocessing differ from the historical workflow.

File-based MLflow uses `./mlruns`. On PowerShell set `$env:MLFLOW_ALLOW_FILE_STORE = "true"` or on Linux run `export MLFLOW_ALLOW_FILE_STORE=true`. The training and experiment commands also default this variable to true. If Windows Application Control blocks a compiled dependency, keep the policy enabled and use project-local training without the MLflow flag, or use Codespaces for tracking.

### Database-backed MLflow in Codespaces

Open a Codespace for this branch. In one terminal start the server, then open the forwarded private port 5000 for the dashboard.

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 0.0.0.0 --port 5000
```

In another terminal:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m src.mp083.train --mlflow --registry-name mp083-random-forest
```

SQLite supports MLflow Model Registry versions. The local serving manifest is a separate filesystem fallback and remains the approval source for the API. Preserve the MLflow database and artifact directory if the run history must outlive the Codespace.

## Approve, select and roll back

Review `reports/training.json` before explicitly approving the first candidate. Use the version printed by training or stored in the report.

```bash
python -m src.mp083.registry show <version>
python -m src.mp083.registry approve <version>
python -m src.mp083.registry show
python -m src.mp083.registry rollback <previous-version>
```

Use real version IDs in place of angle-bracket placeholders. Each version has its own pipeline and metadata directory. Retain prior version directories for rollback. Set `MODEL_VERSION` to load a specific version, or leave it unset to use `approved.json`. Restart serving workers after a change. Only load trusted joblib artifacts, using the pinned scikit-learn version.

## Serve and predict

```bash
python -m uvicorn app.main:app --port 8000 --no-access-log --no-use-colors
```

`/` describes the service, `/docs` contains OpenAPI, `/live` checks process responsiveness, `/health` checks model readiness, and `/model-info` shows the loaded model version and evaluation metadata. Startup failures leave the process live with readiness returning HTTP 503. Invalid prediction requests return HTTP 422.

Linux:

```bash
curl -sS http://localhost:8000/predict -H 'Content-Type: application/json' -d '{"age":25,"screen_hours":6,"apps_installed":42,"plan_gb_limit":3,"video_minutes":75,"wifi_hours":4,"phone_tier":"mid","circle":"north"}'
```

PowerShell:

```powershell
$body = @{age=25; screen_hours=6; apps_installed=42; plan_gb_limit=3; video_minutes=75; wifi_hours=4; phone_tier="mid"; circle="north"} | ConvertTo-Json
Invoke-RestMethod http://localhost:8000/predict -Method Post -ContentType application/json -Body $body
```

The response includes `gb_per_day` and the model version. Application events carry JSON messages with a timestamp, level and event name. Request completions include a request ID, duration, HTTP status and prediction status where applicable. Client bodies and raw error text are excluded.

## Docker and Kubernetes

Build the image, then mount the complete model store read-only. On Linux, make the host path readable by UID 10001.

```bash
docker build -t g174-mp083-api .
docker run --rm -p 8000:8000 --mount "type=bind,source=${PWD}/models,target=/service/models,readonly" g174-mp083-api
```

In PowerShell the same build command works. Use this mount syntax:

```powershell
docker run --rm -p 8000:8000 --mount "type=bind,source=$($PWD.Path)/models,target=/service/models,readonly" g174-mp083-api
```

For Kubernetes, create a `mp083-models` PVC in the deployment namespace using the cluster's storage class and provision it with `approved.json` and the corresponding version directory. The checked-in image tag refers to a local build. To deploy from a registry, set `IMAGE` below to the real destination, publish the image and replace `image:` in `k8s/deployment.yaml` with that same reference. Give the cluster pull access if needed.

```bash
docker tag g174-mp083-api "$IMAGE"
docker push "$IMAGE"
kubectl apply -f k8s/
kubectl rollout status deployment/mp083-api
kubectl get pods,pvc
kubectl port-forward service/mp083-api 8000:8000
```

Check `kubectl describe pod <pod>` for scheduling, pull and volume errors, and `kubectl logs deployment/mp083-api` for model startup errors. Confirm file permissions and the approval pointer if readiness returns 503. Restart the deployment after changing the mounted approval pointer.

## Monitor drift and group performance

```bash
python -m src.mp083.monitoring
```

The numeric Kolmogorov–Smirnov distance compares empirical distributions. Categorical total variation measures how much probability mass changes across labels. The report also measures changes in missing or invalid-value rates after normalization. Thresholds are in `params.yaml`. Use `--reference` to provide the actual training reference when available. The default uses the raw training pool. Small samples, seasonality and changing group mixtures can affect the distances. A flag does not establish accuracy loss, and feature drift cannot measure label or concept drift by itself.

Training reports held-out MAE, RMSE and R² for each `phone_tier` and `circle` group meeting the minimum row count. Group performance gaps can reflect sample size or different usage distributions and do not establish unfairness by themselves. These features do not identify sensitive demographic characteristics.

## Retrain with a promotion gate

Point a copy of `params.yaml` at the new training CSV. Provide a separate, fresh labeled evaluation CSV with enough rows. It must be independent of both models' training data. Exact profile and record-ID checks detect overlap when those keys are available. Repeatedly using the same evaluation set for tuning can bias the comparison.

```bash
python -m src.mp083.train --config retrain.yaml --evaluation-data path/to/evaluation.csv --promote
```

The command evaluates candidate and incumbent on the same rows. It saves a candidate and the comparison and only updates the approval pointer when configured MAE and RMSE criteria pass. Omit `--promote` to review results before changing the pointer.

## Score holdout and run tests

```bash
python -m src.mp083.holdout
python -m ruff check .
python -m pytest -q
```

Holdout scoring uses `data/raw/holdout_features.csv` and the approved model. Predictions are written to `predictions/holdout_predictions.csv` in input order with the original record IDs when present. `reports/holdout_rejections.json` lists the one-based data row numbers and invalid fields for rows rejected under the API schema.

GitHub Actions runs Ruff and pytest on pushes and pull requests, using generated test fixtures. Use feature branches and review changes through pull requests. Keep raw data, virtual environments, generated model artifacts and experiment databases out of Git. Update DVC pointer files together with authorized dataset changes, and share the matching data through the team's configured DVC remote.

See [final verification](reports/final_verification.md) and [completion summary](reports/project_completion_summary.md) for results and open items.

If the build network uses a private certificate authority, supply its trusted CA bundle to BuildKit with `docker build --secret id=build_ca,src=/path/to/ca-bundle.crt -t g174-mp083-api .`.
