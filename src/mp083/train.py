import pandas as pd
import numpy as np
import yaml
import joblib
import mlflow

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# STEP 1: LOAD CONFIGURATION
# ==========================================

with open("params.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# ==========================================
# STEP 2: LOAD CLEANED DATA
# ==========================================

df = pd.read_csv("data/processed/train_clean.csv")

print("Dataset loaded:", df.shape)

# ==========================================
# STEP 3: SELECT FEATURES AND TARGET
# ==========================================

numeric_features = config["features"]["numeric"]
categorical_features = config["features"]["categorical"]

features = numeric_features + categorical_features

target = config["target"]

X = df[features]
y = pd.to_numeric(df[target], errors="coerce")

# Exclude rows without valid targets
valid_rows = y.notna() & np.isfinite(y)

X = X.loc[valid_rows]
y = y.loc[valid_rows]

print("Features:", features)
print("Target:", target)

# ==========================================
# STEP 4: TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config["model"]["test_size"],
    random_state=config["model"]["random_state"]
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# ==========================================
# STEP 5: PREPROCESSING
# ==========================================

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(
        strategy="constant",
        fill_value="unknown"
    )),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
])

# ==========================================
# STEP 6: RANDOM FOREST MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=config["model"]["random_state"],
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# ==========================================
# STEP 7: CONFIGURE MLFLOW
# ==========================================

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

mlflow.set_experiment(
    config["mlflow"]["experiment_name"]
)

# ==========================================
# STEP 8: TRAIN AND TRACK EXPERIMENT
# ==========================================

with mlflow.start_run(run_name="random_forest_baseline"):

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    # Log model parameters
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", config["model"]["random_state"])

    # Log evaluation metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    # Save trained pipeline locally
    Path("models").mkdir(exist_ok=True)

    model_path = "models/random_forest.joblib"

    joblib.dump(pipeline, model_path)

    mlflow.log_artifact(model_path)

    print("\n========== MODEL TRAINING COMPLETED ==========")

    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))

    print("\nModel saved to:", model_path)

    print("MLflow Run ID:", mlflow.active_run().info.run_id)