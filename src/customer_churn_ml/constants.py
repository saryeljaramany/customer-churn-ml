"""Domain constants for the customer churn project."""

from __future__ import annotations

from typing import Final

ID_COL: Final[str] = "customerID"
TARGET_COL: Final[str] = "Churn"
TARGET_ENCODED_COL: Final[str] = "Churn_Yes"

NUMERIC_COLS: Final[tuple[str, ...]] = (
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
)

CONTRACT_ORDER: Final[tuple[str, ...]] = (
    "Month-to-month",
    "One year",
    "Two year",
)

INTERNET_ORDER: Final[tuple[str, ...]] = (
    "No",
    "DSL",
    "Fiber optic",
)

MISSING_CATEGORY: Final[str] = "__MISSING__"
