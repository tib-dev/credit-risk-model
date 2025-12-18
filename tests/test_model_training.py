"""
Pytest tests for model training pipeline with MLflow mocked.
"""

import pandas as pd
import numpy as np
import pytest

from sklearn.model_selection import train_test_split

from credit_risk_model.models.train import train_and_tune_models


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_data():
    """
    Generate a small synthetic dataset for testing.
    """
    rng = np.random.default_rng(42)

    n = 200
    df = pd.DataFrame(
        {
            "feature_1": rng.normal(size=n),
            "feature_2": rng.uniform(0, 1, size=n),
            "feature_3": rng.integers(0, 5, size=n),
            "is_high_risk": rng.integers(0, 2, size=n),
        }
    )
    return df


@pytest.fixture(scope="session")
def train_test_data(sample_data):
    """
    Train-test split for synthetic dataset.
    """
    X = sample_data.drop(columns=["is_high_risk"])
    y = sample_data["is_high_risk"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


@pytest.fixture(autouse=True)
def mock_mlflow(monkeypatch):
    """
    Mock MLflow functions to avoid filesystem and registry side effects.
    """

    class DummyRun:
        info = type("info", (), {"run_id": "test_run_id"})

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    monkeypatch.setattr("mlflow.start_run", lambda *a, **k: DummyRun())
    monkeypatch.setattr("mlflow.log_params", lambda *a, **k: None)
    monkeypatch.setattr("mlflow.log_metrics", lambda *a, **k: None)
    monkeypatch.setattr("mlflow.log_metric", lambda *a, **k: None)
    monkeypatch.setattr("mlflow.sklearn.log_model", lambda *a, **k: None)
    monkeypatch.setattr("mlflow.set_tag", lambda *a, **k: None)
    monkeypatch.setattr("mlflow.active_run", lambda: DummyRun())


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_training_pipeline_runs(train_test_data):
    """
    Training pipeline should run without errors and return models.
    """
    X_train, X_test, y_train, y_test = train_test_data

    trained_models = train_and_tune_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    assert isinstance(trained_models, dict)
    assert len(trained_models) > 0


def test_models_have_expected_keys(train_test_data):
    """
    Each trained model must return required metadata.
    """
    X_train, X_test, y_train, y_test = train_test_data

    trained_models = train_and_tune_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    for name, info in trained_models.items():
        assert "model" in info
        assert "metrics" in info
        assert "run_id" in info


def test_models_have_required_metrics(train_test_data):
    """
    Ensure all required evaluation metrics are present.
    """
    X_train, X_test, y_train, y_test = train_test_data

    trained_models = train_and_tune_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    required_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    }

    for info in trained_models.values():
        assert required_metrics.issubset(info["metrics"].keys())


def test_models_are_fitted(train_test_data):
    """
    Returned estimators should be fitted and usable.
    """
    X_train, X_test, y_train, y_test = train_test_data

    trained_models = train_and_tune_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    for info in trained_models.values():
        model = info["model"]
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")
