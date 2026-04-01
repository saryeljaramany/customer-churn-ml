"""Inference script for customer churn predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .config import PATHS
from .constants import NUMERIC_COLS
from .utils import get_logger, load_pickle

logger = get_logger(__name__)


def load_model_artifacts(model_dir: Path | None = None) -> tuple:
    """Load the trained model, feature names, and preprocessor."""
    if model_dir is None:
        model_dir = PATHS.model_dir
    else:
        model_dir = Path(model_dir)  # Convert string to Path

    model_path = model_dir / "churn_model.pkl"
    feature_names_path = model_dir / "feature_names.pkl"
    preprocessor_path = model_dir / "preprocessor.pkl"

    # Verify all files exist
    for path in [model_path, feature_names_path, preprocessor_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    # Load artifacts
    model = load_pickle(model_path)
    feature_names = load_pickle(feature_names_path)
    preprocessor = load_pickle(preprocessor_path)

    logger.info("Loaded model from %s", model_path)
    logger.info("Loaded %d features", len(feature_names))
    logger.info("Loaded preprocessor (fitted)")

    return model, feature_names, preprocessor


def predict_churn(
    input_path: str | Path,
    output_path: str | Path | None = None,
    model_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Predict customer churn from raw input data.

    Args:
        input_path: Path to CSV with raw customer data (same format as training)
        output_path: Optional path to save predictions CSV
        model_dir: Directory containing model artifacts (default: model/)

    Returns:
        DataFrame with columns: customer_id, churn_probability, confidence
    """
    input_path = Path(input_path)

    # 1. Load input data
    logger.info("Loading input data from %s", input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    logger.info("Input shape: %s", df.shape)

    # Validate required ID column
    id_col = PATHS.id_col.name if hasattr(PATHS.id_col, "name") else str(PATHS.id_col)
    if id_col not in df.columns:
        raise ValueError(f"Input must contain '{id_col}' column. Found columns: {list(df.columns)}")

    # 2. Load model artifacts
    model, feature_names, preprocessor = load_model_artifacts(model_dir)

    # 3. Preprocess the input data
    try:
        X_processed = preprocessor.transform(df)
    except Exception as e:
        logger.error("Preprocessing failed: %s", e)
        raise

    # 4. Scale numeric features
    model_dir_resolved = Path(model_dir) if model_dir else PATHS.model_dir
    scaler_path = model_dir_resolved / "scaler.pkl"
    if scaler_path.exists():
        scaler = load_pickle(scaler_path)
        numeric_cols_present = [col for col in NUMERIC_COLS if col in X_processed.columns]
        if numeric_cols_present:
            scaled_array = scaler.transform(X_processed[numeric_cols_present])
            X_processed[numeric_cols_present] = scaled_array
            logger.info("Applied scaling to numeric columns: %s", numeric_cols_present)
        else:
            logger.warning("No numeric columns found for scaling")
    else:
        logger.warning("Scaler not found at %s. Skipping scaling.", scaler_path)

    # 5. Ensure we have exactly the features the model expects
    missing_features = set(feature_names) - set(X_processed.columns)
    if missing_features:
        logger.warning("Missing features (adding with 0): %s", missing_features)
        for feat in missing_features:
            X_processed[feat] = 0

    # 6. Keep only model features in correct order
    X = X_processed[feature_names]
    logger.info("Model input shape: %s", X.shape)

    # 7. Generate predictions
    logger.info("Generating predictions...")
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        # Fallback for models without predict_proba
        probabilities = model.predict(X)

    # 5. Create output DataFrame
    customer_ids = df[id_col].values
    confidence = probabilities.copy()

    results = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "churn_probability": probabilities.round(6),
            "confidence": confidence.round(6),
        }
    )

    logger.info("Predictions generated for %d customers", len(results))

    # 6. Save results if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        logger.info("Predictions saved to %s", output_path)

    return results


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Predict customer churn using trained model")
    parser.add_argument(
        "-i", "--input", required=True, help="Input CSV file with raw customer data"
    )
    parser.add_argument(
        "-o", "--output", required=False, help="Output CSV file path for predictions"
    )
    parser.add_argument(
        "--model-dir",
        required=False,
        default=None,
        help="Directory containing model artifacts (default: model/)",
    )

    args = parser.parse_args()

    try:
        results = predict_churn(
            input_path=args.input, output_path=args.output, model_dir=args.model_dir
        )

        # Print summary to stdout
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        print(f"Total customers: {len(results)}")
        print(f"Mean churn probability: {results['churn_probability'].mean():.2%}")
        print(f"High risk (prob > 50%): {(results['churn_probability'] > 0.5).sum()} customers")
        print("\nFirst 10 predictions:")
        print(results.head(10).to_string(index=False))

        return 0

    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
