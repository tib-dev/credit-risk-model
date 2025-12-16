# scripts/run_features.py

import argparse
import pandas as pd
from credit_risk_model.core.settings import settings
from credit_risk_model.data.load_data import DataLoader
from credit_risk_model.data.pipeline import run_feature_engineering


def main(file: str):
    # -------------------------
    # Paths
    # -------------------------
    interim_path = settings.paths.data["interim_dir"] / file
    processed_path = settings.paths.data["processed_dir"]

    # -------------------------
    # Load Data
    # -------------------------
    loader = DataLoader(filepath=interim_path, file_type="csv")
    df = loader.load()
    print(f"[INFO] Loaded data shape: {df.shape}")

    # -------------------------
    # Load Features from Settings
    # -------------------------
    NUMERIC_FEATURES = settings.config.get("numeric_features", [])
    CATEGORICAL_FEATURES = settings.config.get("categorical_features", [])

    # -------------------------
    # Feature Engineering
    # -------------------------
    X_processed, pipeline = run_feature_engineering(
        df=df,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        use_woe=False
    )

    # -------------------------
    # Get Feature Names
    # -------------------------
    try:
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out(
        )
    except AttributeError:
        feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    # -------------------------
    # Convert to DataFrame
    # -------------------------
    X_df = pd.DataFrame(
        X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed,
        columns=feature_names,
        index=df.index
    )

    # -------------------------
    # Save Processed Features
    # -------------------------
    output_file = processed_path / "features.parquet"
    X_df.to_parquet(output_file, index=False)
    print(f"[INFO] Features saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run feature engineering pipeline.")
    parser.add_argument("--file", type=str, required=True,
                        help="Input CSV filename in interim_dir")
    args = parser.parse_args()
    main(args.file)
