# Modularization Blueprint: Customer Churn Prediction

**Purpose:** Transition the research-style Jupyter notebook into a production-grade `src/` package while keeping the notebook for experimentation.

**Scope:** Analysis only — no implementation code. `predict.py` will be implemented in a separate phase.

---

## 1. Proposed `src/` Directory Structure

```
customer_churn_ml/
├── pyproject.toml                    # Package metadata, dependencies, editable install
├── README.md
├── data/
│   ├── raw/                          # Raw input data (unchanged)
│   └── processed/                    # Cleaned data, scaler, feature names
├── model/                            # Saved model artifacts
├── notebooks/
│   └── Customer_Churn_Project.ipynb  # Slim notebook importing from src/
└── src/
    └── customer_churn_ml/
        ├── __init__.py               # Package root; exposes public API
        ├── config.py                 # Paths, hyperparameters, constants (single source of truth)
        ├── constants.py              # Domain constants (column names, target label, numeric cols, etc.)
        ├── data/
        │   ├── __init__.py
        │   └── loader.py             # Load raw or processed CSV; return DataFrame
        ├── preprocessing/
        │   ├── __init__.py
        │   ├── preprocessor.py       # End-to-end preprocessing pipeline (clean, encode, no scaling)
        │   └── scaler.py             # Fit/transform numeric cols; save/load scaler
        ├── training/
        │   ├── __init__.py
        │   ├── trainer.py            # Train models, evaluate, select best, save artifacts
        │   └── evaluator.py          # Accuracy, ROC-AUC, confusion matrix, classification report, ROC curves
        ├── interpret/
        │   ├── __init__.py
        │   └── feature_importance.py # Extract and format feature importances or coefficients
        ├── utils.py                  # Generic helpers (logging setup, path resolution, validation)
        └── predict.py                # Inference script (implemented in separate phase)
```

### File Descriptions

| File | Single Responsibility |
|------|------------------------|
| `__init__.py` | Expose `load_data`, `preprocess`, `train`, etc. for notebook imports |
| `config.py` | Centralize paths, `RANDOM_STATE`, `CARDINALITY_THRESHOLD`, `test_size`, model hyperparameters |
| `constants.py` | Define `TARGET_COL`, `ID_COL`, `NUMERIC_COLS`, `CONTRACT_ORDER`, etc. |
| `data/loader.py` | Load raw or processed CSV; validate columns; return `pd.DataFrame` |
| `preprocessing/preprocessor.py` | Drop ID, fix TotalCharges, encode target, encode categoricals, save cleaned CSV |
| `preprocessing/scaler.py` | Fit scaler on train, transform train/test; save/load scaler pickle |
| `training/trainer.py` | Build models, fit, call evaluator, select best by ROC-AUC, save model + feature names |
| `training/evaluator.py` | Compute metrics, confusion matrix, classification report; plot ROC curves; return results dict |
| `interpret/feature_importance.py` | Extract importances/coefficients from model; return DataFrame for plotting |
| `utils.py` | Logging setup, `resolve_path`, validation helpers, duplicate check logic |

---

## 2. Cell-to-Module Mapping

| Cell # | Block Name | Target Module | Refactoring Action | Reason |
|--------|------------|---------------|--------------------|--------|
| 0 | Title / intro | — | Keep as markdown in notebook | Documentation only |
| 1 | Load Libraries | `notebooks` (imports only) | Replace with `from customer_churn_ml import ...` | Imports come from package |
| 2 | Imports (pandas, sklearn, pickle, sns) | `config.py` + package `__init__` | Move sklearn/pandas to package deps; notebook imports package | Centralize dependencies |
| 3 | Load Dataset header | — | Keep as markdown | Section header |
| 4 | Load raw CSV | `data/loader.py` | `load_raw_data(path) -> pd.DataFrame`; notebook calls it | Reusable, testable I/O |
| 5–6 | EDA intro, Data Overview | **Stay in notebook** | Keep in notebook; optionally call `utils.check_integrity(df)` | EDA is exploratory; integrity check can be a util |
| 7 | Data Overview + Integrity Checks | `utils.py` | `check_data_integrity(df) -> dict`; notebook calls and prints | Reusable validation logic |
| 8 | df.describe() | **Stay in notebook** | Keep as-is | One-line exploratory; not worth extracting |
| 9–20 | EDA visualizations | **Stay in notebook** | Keep all plotting in notebook | EDA plots are iterative, stakeholder-facing; not production code |
| 21–22 | Preprocessing header | — | Keep as markdown | Section header |
| 23 | Full preprocessing pipeline | `preprocessing/preprocessor.py` | `Preprocessor(config).fit_transform(df) -> pd.DataFrame`; save CSV inside or via config | Single responsibility; testable; configurable |
| 24 | Split Data header | — | Keep as markdown | Section header |
| 25 | Load processed, split, scale | `data/loader.py`, `preprocessing/scaler.py`, `training/trainer` or notebook | `load_processed_data()`; `split_data(X, y)`; `Scaler.fit_transform(X_train, X_test)`; orchestrate in notebook or trainer | I/O, scaling, and split are separate concerns |
| 26 | Train Models header | — | Keep as markdown | Section header |
| 27 | Training, evaluation, ROC, comparison, save | `training/trainer.py`, `training/evaluator.py` | `Trainer(config).train(X_train, y_train, X_test, y_test)`; `Evaluator.evaluate()`; save model + feature_names in trainer | Separation of training vs evaluation; testable |
| 28–29 | Feature Importance | `interpret/feature_importance.py` | `get_feature_importance_df(model, feature_names, model_type)`; notebook calls and plots | Pure logic; plotting stays in notebook |
| 30–31 | Model Saved | — | Simplify to markdown + optional verification util | Documentation; verification is trivial |

### Cells That Must Stay in Notebook Only

| Cell(s) | Reason |
|---------|--------|
| 5–20 (EDA section) | Exploratory visualizations are iterative, presentation-focused; moving to package would over-engineer and reduce flexibility for ad-hoc analysis. |
| 8 (`df.describe()`) | One-liner exploratory; no reusable logic. |
| All `plt.show()` / `sns.*` plotting | Visualization is notebook's job; package returns data (DataFrames, metrics dicts) and notebook renders. |

---

## 3. Notebook Adaptation Plan

### 3.1 Package Setup and Imports

1. **Add `pyproject.toml`** at project root with:
   - `[project]` name, version, dependencies
   - `[build-system]` (setuptools or hatch)
   - `[tool.setuptools.packages.find]` with `where = ["src"]`
   - Install as editable: `pip install -e .`

2. **Alternative:** Use `sys.path` in the first notebook cell:
   ```python
   import sys
   sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
   ```
   Editable install is preferred for reproducibility and cleaner imports.

3. **Notebook imports** (replace cell 2):
   ```python
   from customer_churn_ml import load_raw_data, preprocess, train, get_feature_importance
   from customer_churn_ml.config import PATHS, RANDOM_STATE
   ```

### 3.2 `if __name__ == "__main__"` Guards

| Module | Guard Needed? | Reason |
|--------|---------------|--------|
| `config.py` | No | Pure config; no executable script |
| `data/loader.py` | Optional | If run as script for ad-hoc load test |
| `preprocessing/preprocessor.py` | Yes | CLI entry point for preprocessing-only runs (e.g., `python -m customer_churn_ml.preprocessing.preprocessor`) |
| `preprocessing/scaler.py` | No | Called by trainer/notebook |
| `training/trainer.py` | Yes | CLI entry point for training-only runs |
| `interpret/feature_importance.py` | No | Pure function; no side effects |
| `utils.py` | No | Library code |
| `predict.py` | Yes | Inference script; always has `if __name__ == "__main__"` |

### 3.3 Configuration Externalization

Lift into `config.py` (or `.env` + `config.py`):

| Value | Current Location | Target |
|-------|------------------|--------|
| Raw data path | Cell 4 | `config.PATHS.raw_data` |
| Processed data path | Cells 23, 25 | `config.PATHS.processed_data` |
| Model directory | Cells 23, 25, 27 | `config.PATHS.model_dir` |
| `RANDOM_STATE` | Cells 23, 27 | `config.RANDOM_STATE` |
| `CARDINALITY_THRESHOLD` | Cell 23 | `config.CARDINALITY_THRESHOLD` |
| `test_size` | Cell 25 | `config.TEST_SIZE` |
| `save_splits` | Cell 23 | `config.SAVE_SPLITS` |
| Model hyperparameters (max_iter, n_estimators) | Cell 27 | `config.MODEL_PARAMS` |

Lift into `constants.py`:

| Value | Target |
|-------|--------|
| `customerID`, `Churn`, `Churn_Yes` | `ID_COL`, `TARGET_COL`, `TARGET_ENCODED` |
| `['tenure', 'MonthlyCharges', 'TotalCharges']` | `NUMERIC_COLS` |
| Contract/Internet ordering | `CONTRACT_ORDER`, `INTERNET_ORDER` |

---

## 4. Production-Readiness Checklist (Per Module)

### 4.1 `config.py`

| Item | Applies? | How to Address |
|------|----------|----------------|
| Type hints | Yes | Add type annotations for all config classes/attrs |
| Module-level logging | No | Config rarely logs; optional for path resolution |
| Input validation | Yes | Validate paths exist (or are creatable); validate numeric ranges |
| I/O vs logic separation | N/A | Config is data only |
| Externalized config | Yes | All paths and constants come from here |
| Testability | Yes | Config should be injectable (e.g., override in tests) |

### 4.2 `constants.py`

| Item | Applies? | How to Address |
|------|----------|----------------|
| Type hints | Yes | Use `typing.Final` or `Literal` where appropriate |
| Logging | No | Constants only |
| Input validation | No | Constants are fixed |
| I/O separation | N/A | No I/O |
| Externalized config | Yes | Centralized constants |
| Testability | Yes | Import and assert values in tests |

### 4.3 `data/loader.py`

| Item | Applies? | How to Address |
|------|----------|----------------|
| Type hints | Yes | `def load_raw_data(path: Path | str) -> pd.DataFrame` |
| Module-level logging | Yes | Log load success, row count, columns |
| Input validation | Yes | Validate path exists; validate required columns present |
| I/O vs logic separation | Yes | Load only; no business logic |
| Externalized config | Yes | Path from config |
| Testability | Yes | Use fixtures or temp files in tests |

### 4.4 `preprocessing/preprocessor.py`

| Item | Applies? | How to Address |
|------|----------|----------------|
| Type hints | Yes | All public methods; `fit_transform(df: pd.DataFrame) -> pd.DataFrame` |
| Module-level logging | Yes | Replace prints with `logger.info` |
| Input validation | Yes | Validate required columns; handle missing Churn |
| I/O vs logic separation | Yes | Save to disk in a separate step or via config; core logic pure |
| Externalized config | Yes | Use config + constants |
| Testability | Yes | Pass DataFrame in/out; mock file writes |

### 4.5 `preprocessing/scaler.py`

| Item | Applies? | How to Address |
|------|----------|----------------|
| Type hints | Yes | `fit_transform(X_train, X_test, cols)` etc. |
| Module-level logging | Yes | Log fit/transform; log save path |
| Input validation | Yes | Validate cols exist in DataFrame |
| I/O vs logic separation | Yes | Separate `save_scaler()` from `fit_transform()` |
| Externalized config | Yes | Paths and column list from config/constants |
| Testability | Yes | Pure fit/transform logic; mock pickle dump |

### 4.6 `training/trainer.py`

| Item | Applies? | How to Address |
|------|----------|----------------|
| Type hints | Yes | Return typed results dict; model type hints |
| Module-level logging | Yes | Replace all prints with logger |
| Input validation | Yes | Validate X_train/X_test shapes; validate y |
| I/O vs logic separation | Yes | Separate `save_model()`, `save_feature_names()` from train loop |
| Externalized config | Yes | Model params, paths from config |
| Testability | Yes | Inject evaluator; mock file writes |

### 4.7 `training/evaluator.py`

| Item | Applies? | How to Address |
|------|----------|----------------|
| Type hints | Yes | `evaluate(y_true, y_pred, y_prob, model_name) -> dict` |
| Module-level logging | Yes | Log metrics |
| Input validation | Yes | Validate array shapes; handle empty arrays |
| I/O vs logic separation | Yes | Return data only; plotting in notebook or separate viz module |
| Externalized config | No | Evaluation logic is stateless |
| Testability | Yes | Pure functions; no I/O |

### 4.8 `interpret/feature_importance.py`

| Item | Applies? | How to Address |
|------|----------|----------------|
| Type hints | Yes | `get_feature_importance_df(model, feature_names, model_type) -> pd.DataFrame` |
| Module-level logging | Optional | Log when unknown model type |
| Input validation | Yes | Validate model has `feature_importances_` or `coef_` |
| I/O separation | N/A | No I/O |
| Externalized config | No | Pure transformation |
| Testability | Yes | Pass mock model; assert DataFrame shape/columns |

### 4.9 `utils.py`

| Item | Applies? | How to Address |
|------|----------|----------------|
| Type hints | Yes | All helper functions |
| Module-level logging | Yes | Use logger for warnings |
| Input validation | Yes | In `check_data_integrity` etc. |
| I/O separation | N/A | Utils are logic only |
| Externalized config | Partial | Path resolution may use config |
| Testability | Yes | Pure or injectable |

### 4.10 `predict.py` (Future Phase)

| Item | Applies? | How to Address |
|------|----------|----------------|
| Type hints | Yes | Full typing for inference pipeline |
| Module-level logging | Yes | Log load, preprocess, predict |
| Input validation | Yes | Validate input schema; feature alignment |
| I/O separation | Yes | Load model/features; predict; optionally save outputs |
| Externalized config | Yes | Model path, scaler path from config |
| Testability | Yes | Unit tests with saved fixtures |

---

## Summary

- **Extract:** Data loading, preprocessing, scaling, training, evaluation logic, and feature importance computation into `src/`.
- **Keep in notebook:** EDA visualizations, `df.describe()`, orchestration cells, and all plotting.
- **Externalize:** Paths, hyperparameters, and constants into `config.py` and `constants.py`.
- **Standardize:** Type hints, logging, validation, and testability across all new modules.
