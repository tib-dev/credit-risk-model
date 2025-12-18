"""
Run end-to-end model training, tuning, evaluation, and registration.

Usage:
    python -m credit_risk_model.run_model
"""

import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split

from credit_risk_model.core.settings import settings
from credit_risk_model.utils.project_root import get_project_root
from credit_risk_model.data.load_data import DataLoader
from credit_risk_model.models.train import train_and_tune_models
from credit_risk_model.models.register import register_model


def main():
    """
    Main training entry point.
    """

    # ------------------------------------------------------------------
    # MLflow setup (force tracking at project root)
    # ------------------------------------------------------------------
    root = get_project_root()
    tracking_path = (root / "mlruns").as_posix()

    mlflow.set_tracking_uri(f"file:///{tracking_path}")
    mlflow.set_experiment("credit-risk-model")

    # ------------------------------------------------------------------
    # Load processed dataset
    # ------------------------------------------------------------------
    feature_file = "features_with_target.parquet"
    processed_path = settings.paths.data["processed_dir"] / feature_file

    loader = DataLoader(filepath=processed_path, file_type="parquet")
    df = loader.load()

    print(f"Loaded dataset shape: {df.shape}")

    # ------------------------------------------------------------------
    # Prepare features and target
    # ------------------------------------------------------------------
    TARGET_COL = "is_high_risk"

    # Remove identifiers (not predictive)
    df = df.drop(columns=["CustomerId"], errors="ignore")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # ------------------------------------------------------------------
    # Train, tune, and track models
    # ------------------------------------------------------------------
    trained_models = train_and_tune_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    # ------------------------------------------------------------------
    # Compare results
    # ------------------------------------------------------------------
    results = []

    for name, info in trained_models.items():
        row = {"model": name}
        row.update(info["metrics"])
        results.append(row)

    results_df = pd.DataFrame(results).sort_values(
        by="roc_auc", ascending=False
    )

    print("\nModel comparison:")
    print(results_df)

    # ------------------------------------------------------------------
    # Select best model
    # ------------------------------------------------------------------
    best_model_name, best_info = max(
        trained_models.items(),
        key=lambda x: x[1]["metrics"]["roc_auc"],
    )

    print(f"\nBest model: {best_model_name}")
    print(f"Metrics: {best_info['metrics']}")

    # ------------------------------------------------------------------
    # Register best model
    # ------------------------------------------------------------------
    register_model(
        model_name="credit_risk_model",
        run_id=best_info["run_id"],
    )

    print("\nModel registered successfully.")
    print("Run `mlflow ui` to inspect experiments.")


if __name__ == "__main__":
    main()
