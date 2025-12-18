"""
Model factory for supported algorithms.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def build_model(name: str, params: dict):
    """
    Instantiate a model based on name and parameters.
    """

    if name == "logistic_regression":
        return LogisticRegression(**params)

    if name == "random_forest":
        return RandomForestClassifier(**params)

    if name == "xgboost":
        return xgb.XGBClassifier(
            use_label_encoder=False,
            **params,
        )

    raise ValueError(f"Unsupported algorithm: {name}")
