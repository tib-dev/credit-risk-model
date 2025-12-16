from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from credit_risk_model.features.aggregate import CustomerAggregates
from credit_risk_model.features.datetime import DatetimeFeatures
from credit_risk_model.features.woe_iv import WoETransformer


def build_feature_pipeline(
    numeric_features,
    categorical_features,
    use_woe=False,
    woe_features=None,
):
    """
    Build an end-to-end sklearn feature engineering pipeline.

    The pipeline performs:
    - Customer-level aggregations
    - Datetime feature extraction
    - Missing value imputation
    - Categorical encoding
    - Feature scaling
    - Optional WoE transformation for scorecards

    Parameters
    ----------
    numeric_features : list[str]
        Names of numerical features to be scaled.
    categorical_features : list[str]
        Names of categorical features to be encoded.
    use_woe : bool, default=False
        Whether to apply Weight of Evidence transformation.
    woe_features : list[str], optional
        Features to be transformed using WoE (required if use_woe=True).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fully configured feature engineering pipeline.
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    steps = [
        ("aggregates", CustomerAggregates()),
        ("datetime", DatetimeFeatures()),
    ]

    if use_woe:
        if not woe_features:
            raise ValueError("woe_features must be provided when use_woe=True")
        steps.append(("woe", WoETransformer(woe_features)))

    steps.append(("preprocess", preprocessor))

    return Pipeline(steps)
