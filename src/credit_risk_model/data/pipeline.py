"""
Data pipeline orchestration for feature engineering.

This module connects raw transaction data with the feature engineering
pipeline, producing a fully transformed, model-ready dataset.
"""

import pandas as pd
from typing import Tuple, Any

from credit_risk_model.features.feature_pipeline import build_feature_pipeline


def run_feature_engineering(
    df: pd.DataFrame,
    numeric_features: list,
    categorical_features: list,
    use_woe: bool = False,
    woe_features: list | None = None,
) -> Tuple[Any, Any]:
    """
    Execute the full feature engineering pipeline.

    This function builds and applies the sklearn-based feature pipeline
    that performs aggregation, datetime extraction, missing value handling,
    encoding, scaling, and optional WoE transformation.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw transaction-level dataset.
    numeric_features : list[str]
        Numerical feature names to be scaled.
    categorical_features : list[str]
        Categorical feature names to be encoded.
    use_woe : bool, default=False
        Whether to apply Weight of Evidence transformation.
    woe_features : list[str] or None
        Features to be transformed using WoE (required if use_woe=True).

    Returns
    -------
    X_processed : array-like (usually scipy.sparse.csr_matrix)
        Transformed feature matrix ready for model training.
    pipeline : sklearn Pipeline
        Fitted feature engineering pipeline (used to recover feature names).
    """

    pipeline = build_feature_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        use_woe=use_woe,
        woe_features=woe_features,
    )

    X_processed = pipeline.fit_transform(df)

    return X_processed, pipeline
