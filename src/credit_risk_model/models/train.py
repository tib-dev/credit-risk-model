"""
Model training orchestration using config-driven setup.
"""

import mlflow
import mlflow.sklearn
import logging

from credit_risk_model.core.settings import settings
from credit_risk_model.models.evaluate import evaluate_classifier
from credit_risk_model.models.tuning import tune_model
from credit_risk_model.models.factory import build_model

logger = logging.getLogger(__name__)


def train_and_tune_models(X_train, X_test, y_train, y_test):
    """
    Train, tune, and track multiple models using MLflow.

    Returns
    -------
    dict
        Mapping of model name to trained model info.
    """

    model_cfg = settings.MODEL
    tuning_cfg = settings.TUNING
    optimize_metric = model_cfg["optimize_metric"]

    trained_models = {}

    for algo in model_cfg["algorithms"]:
        logger.info("Training model: %s", algo)

        base_params = model_cfg[algo].copy()

        # Handle XGBoost class imbalance automatically
        if algo == "xgboost" and base_params.get("scale_pos_weight") == "auto":
            base_params["scale_pos_weight"] = (
                (y_train == 0).sum() / (y_train == 1).sum()
            )

        with mlflow.start_run(run_name=f"{algo}_tuned"):
            model = build_model(algo, base_params)

            tuned_model, best_params = tune_model(
                model=model,
                tuning_cfg=tuning_cfg[algo],
                X_train=X_train,
                y_train=y_train,
                optimize_metric=optimize_metric,
            )

            metrics = evaluate_classifier(tuned_model, X_test, y_test)

            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(tuned_model, artifact_path="model")

            trained_models[algo] = {
                "model": tuned_model,
                "metrics": metrics,
                "run_id": mlflow.active_run().info.run_id,
            }

    return trained_models
