"""Feature importance helpers for tree-based models and linear models."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..utils import get_logger

logger = get_logger(__name__)


def get_feature_importance_df(
    model,
    feature_names: Iterable[str],
    *,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Return a normalized feature importance dataframe."""

    names = list(feature_names)
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        importance_df = pd.DataFrame({"Feature": names, "Importance": values})
        importance_df["AbsoluteImportance"] = importance_df["Importance"].abs()
        importance_df = importance_df.sort_values("Importance", ascending=False)
    elif hasattr(model, "coef_"):
        values = np.asarray(model.coef_).ravel().astype(float)
        importance_df = pd.DataFrame({"Feature": names, "Coefficient": values})
        importance_df["AbsoluteImportance"] = importance_df["Coefficient"].abs()
        importance_df = importance_df.sort_values("AbsoluteImportance", ascending=False)
    else:
        raise ValueError(
            "The supplied model does not expose 'feature_importances_' or 'coef_'."
        )

    if top_n is not None:
        importance_df = importance_df.head(top_n)

    return importance_df.reset_index(drop=True)


def plot_feature_importance(importance_df: pd.DataFrame, *, title: str | None = None, ax=None):
    """Plot a simple horizontal bar chart for an importance dataframe."""

    import matplotlib.pyplot as plt
    import seaborn as sns

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    value_column = "Importance" if "Importance" in importance_df.columns else "Coefficient"
    sns.barplot(data=importance_df, x=value_column, y="Feature", ax=ax)
    ax.set_title(title or "Feature Importance")
    return ax
