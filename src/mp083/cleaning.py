import pandas as pd
import numpy as np
import yaml
from pathlib import Path

# ==========================================
# STEP 1: LOAD CONFIGURATION
# ==========================================

with open("params.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# ==========================================
# STEP 2: LOAD ORIGINAL DATASET
# ==========================================

df = pd.read_csv(config["data"]["train_path"])

print("\n========== DATA CLEANING STARTED ==========")
print("Original dataset shape:", df.shape)

# ==========================================
# STEP 3: REMOVE EXACT DUPLICATES
# ==========================================

duplicate_count = df.duplicated().sum()

print("Exact duplicate rows:", duplicate_count)

df = df.drop_duplicates().copy()

# ==========================================
# STEP 4: CLEAN PHONE TIER SPELLINGS
# ==========================================

def clean_phone_tier(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower().replace(".", "")

    mapping = {
        "basic": "basic",
        "bas": "basic",
        "mid": "mid",
        "flagship": "flagship",
        "fla": "flagship"
    }

    return mapping.get(value, np.nan)


df["phone_tier"] = df["phone_tier"].apply(clean_phone_tier)

print("\nCleaned phone tier categories:")
print(df["phone_tier"].value_counts(dropna=False))

# ==========================================
# STEP 5: HANDLE INVALID AGE VALUES
# ==========================================

df["age"] = pd.to_numeric(df["age"], errors="coerce")

invalid_age_count = (df["age"] == -1).sum()

print("\nInvalid age values (-1):", invalid_age_count)

df.loc[df["age"] == -1, "age"] = np.nan

# ==========================================
# STEP 6: CONVERT NUMERICAL COLUMNS
# ==========================================

numeric_columns = config["features"]["numeric"]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# ==========================================
# STEP 7: HANDLE MISSING NUMERICAL VALUES
# ==========================================

for column in numeric_columns:

    median_value = df[column].median()

    df[column] = df[column].fillna(median_value)

# ==========================================
# STEP 8: HANDLE MISSING CATEGORICAL VALUES
# ==========================================

categorical_columns = config["features"]["categorical"]

for column in categorical_columns:

    df[column] = df[column].fillna("unknown")

# ==========================================
# STEP 9: REMOVE DATA LEAKAGE COLUMN
# ==========================================

if "gb_from_monthly_bill" in df.columns:

    df = df.drop(columns=["gb_from_monthly_bill"])

    print("\nLeakage column removed successfully.")

# ==========================================
# STEP 10: SAVE CLEANED DATASET
# ==========================================

output_path = Path("data/processed/train_clean.csv")

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(output_path, index=False)

print("\n========== DATA CLEANING COMPLETED ==========")

print("Cleaned dataset shape:", df.shape)

print("Saved to:", output_path)

print("\nRemaining missing values:")
print(df.isnull().sum())