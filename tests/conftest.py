"""Shared fixtures for the customer_churn_ml test suite."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Raw data — mimics the Telco CSV structure
# ---------------------------------------------------------------------------


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Minimal raw DataFrame that mirrors the Telco CSV.

    Deliberately includes:
    - An empty TotalCharges string (row index 2) to test coercion + median fill
    - Both Churn=Yes and Churn=No rows
    - All three Contract types and all three InternetService types
    """
    return pd.DataFrame(
        {
            "customerID": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"],
            "tenure": [1, 24, 60, 12, 36, 2, 48, 6],
            "MonthlyCharges": [29.85, 56.95, 89.10, 45.00, 70.50, 35.00, 60.00, 40.00],
            "TotalCharges": [
                "29.85",
                "1365.80",
                "",
                "540.00",
                "2538.00",
                "70.00",
                "2880.00",
                "240.00",
            ],
            "Contract": [
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
                "One year",
                "Month-to-month",
                "Two year",
                "Month-to-month",
            ],
            "InternetService": [
                "DSL",
                "Fiber optic",
                "DSL",
                "No",
                "Fiber optic",
                "DSL",
                "DSL",
                "Fiber optic",
            ],
            "Churn": ["No", "No", "No", "Yes", "No", "Yes", "No", "Yes"],
        }
    )


# ---------------------------------------------------------------------------
# Preprocessed data
# ---------------------------------------------------------------------------


@pytest.fixture
def preprocessed(raw_df):
    """Return (cleaned_df, fitted_preprocessor) — no disk writes."""
    from customer_churn_ml.preprocessing.preprocessor import Preprocessor, PreprocessorConfig

    config = PreprocessorConfig(save_processed_data=False)
    preprocessor = Preprocessor(config)
    df = preprocessor.fit_transform(raw_df)
    return df, preprocessor


@pytest.fixture
def cleaned_df(preprocessed):
    df, _ = preprocessed
    return df


@pytest.fixture
def fitted_preprocessor(preprocessed):
    _, preprocessor = preprocessed
    return preprocessor


# ---------------------------------------------------------------------------
# X / y splits
# ---------------------------------------------------------------------------


@pytest.fixture
def X_y(cleaned_df):
    from customer_churn_ml.constants import TARGET_ENCODED_COL

    X = cleaned_df.drop(columns=[TARGET_ENCODED_COL])
    y = cleaned_df[TARGET_ENCODED_COL]
    return X, y


@pytest.fixture
def split(X_y):
    X, y = X_y
    return train_test_split(X, y, test_size=0.4, random_state=42)


# ---------------------------------------------------------------------------
# Tiny fitted models (fast — 3 estimators)
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted_rf(split):
    X_train, _, y_train, _ = split
    model = RandomForestClassifier(n_estimators=3, random_state=42)
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def fitted_lr(split):
    X_train, _, y_train, _ = split
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    return model
