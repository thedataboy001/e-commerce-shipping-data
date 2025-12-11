import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import List
from src.utils_and_constants import (DROP_COLUMNS, 
                                 NUMERICAL_FEATURES,
                                 CATEGORICAL_FEATURES, PROCESSED_DATA_PATH,
                                 RAW_DATA_PATH, ARTIFACTS_DIR, TARGET_COLUMN)

from sklearn.preprocessing import StandardScaler

ARTIFACTS_DIR = Path(ARTIFACTS_DIR)


def load_data(file_path: str, drop_columns: List[str]) -> pd.DataFrame:
    """Load data from a CSV file."""
    df = pd.read_csv(file_path)
    df.drop(columns=drop_columns, inplace=True, axis=1)
    return df


def preprocess_data_train(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing used during TRAINING.

    - Separates target column (kept as-is)
    - Preprocesses only feature columns
    - Fits and saves scaler
    - Saves feature column names (without target)
    - Returns a dataframe with X_processed + y so you can save it.
    """
    # --- 1. Separate target ---
    y = df[TARGET_COLUMN].copy()
    X = df.drop(columns=[TARGET_COLUMN])

    # --- 2. Numeric conversion on features ---
    X[NUMERICAL_FEATURES] = X[NUMERICAL_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )

    # --- 3. Ordinal mappings ---
    X["Product_importance"] = X["Product_importance"].map(
        {"low": 0, "medium": 1, "high": 2}
    )
    X["Gender"] = X["Gender"].map({"M": 0, "F": 1})

    # --- 4. One-hot encoding on categorical features ---
    X = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=False, dtype=int)

    # --- 5. Fit scaler on training numeric features ---
    scaler = StandardScaler()
    X[NUMERICAL_FEATURES] = scaler.fit_transform(X[NUMERICAL_FEATURES])

    # --- 6. Save artefacts for inference ---
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    feature_columns = X.columns.tolist()  # ✅ NO TARGET INSIDE
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.joblib")
    joblib.dump(feature_columns, ARTIFACTS_DIR / "feature_columns.joblib")

    # --- 7. Return processed features + original target ---
    df_processed = X.copy()
    df_processed[TARGET_COLUMN] = y.values

    return df_processed


def preprocess_data_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing used during INFERENCE (FastAPI / Gradio).

    - Loads scaler + feature_columns
    - Applies same numeric / ordinal / dummy logic
    - Reindexes to training feature columns
    - Does NOT include target column.
    """
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    feature_columns = joblib.load(ARTIFACTS_DIR / "feature_columns.joblib")

    # Drop target if it somehow appears (defensive)
    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])

    # same steps as training
    df[NUMERICAL_FEATURES] = df[NUMERICAL_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    df["Product_importance"] = df["Product_importance"].map(
        {"low": 0, "medium": 1, "high": 2}
    )
    df["Gender"] = df["Gender"].map({"M": 0, "F": 1})

    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=False, dtype=int)

    # Align columns to training feature space (no label here)
    df = df.reindex(columns=feature_columns, fill_value=0)

    # scale numerics using the SAME scaler
    df[NUMERICAL_FEATURES] = scaler.transform(df[NUMERICAL_FEATURES])

    return df


def save_preprocessed_data(df: pd.DataFrame) -> None:
    df.to_csv(PROCESSED_DATA_PATH, index=False)


def main():
    # Load data
    df = load_data(file_path=RAW_DATA_PATH, drop_columns=DROP_COLUMNS)

    # Preprocess data for training (X + y)
    df_processed = preprocess_data_train(df=df)

    # Save preprocessed data (includes class label column)
    save_preprocessed_data(df=df_processed)


if __name__ == "__main__":
    main()