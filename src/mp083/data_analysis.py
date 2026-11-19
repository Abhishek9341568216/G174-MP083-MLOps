import pandas as pd
import yaml

# Load configuration
with open("params.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# Load datasets
train = pd.read_csv(config["data"]["train_path"])
later = pd.read_csv(config["data"]["later_path"])
holdout = pd.read_csv(config["data"]["holdout_path"])

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