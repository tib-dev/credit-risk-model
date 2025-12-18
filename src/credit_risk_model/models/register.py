"""
MLflow model registry utilities.
"""

import mlflow


def register_model(model_name: str, run_id: str, artifact_path: str = "model"):
    """
    Register model from MLflow run into Model Registry.
    """

    model_uri = f"runs:/{run_id}/{artifact_path}"

    mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )
