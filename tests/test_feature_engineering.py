# tests/test_feature_engineering.py

import pytest
import pandas as pd
from pathlib import Path
from credit_risk_model.core.settings import settings
from credit_risk_model.data.load_data import DataLoader
from credit_risk_model.data.pipeline import run_feature_engineering


@pytest.fixture
def sample_data(tmp_path):
    """Create a small sample CSV file for testing."""
    data = {
        "TransactionId": [1, 2, 3],
        "BatchId": [101, 102, 103],
        "AccountId": [1001, 1002, 1003],
        "SubscriptionId": [201, 202, 203],
        "CustomerId": [1, 1, 2],
        "Amount": [100, 150, 200],
        "Value": [10, 15, 20],
        "CurrencyCode": ["USD", "EUR", "USD"],
        "ProviderId": ["P1", "P2", "P1"],
        "ProductId": ["ProdA", "ProdB", "ProdA"],
        "ProductCategory": ["Cat1", "Cat2", "Cat1"],
        "ChannelId": ["Web", "App", "Web"],
        "PricingStrategy": ["Standard", "Promo", "Standard"],
        "TransactionStartTime": ["2025-01-01 10:00:00", "2025-01-01 11:00:00", "2025-01-02 12:00:00"],
        "FraudResult": [0, 1, 0],
    }
    df = pd.DataFrame(data)
    csv_file = tmp_path / "sample.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


def test_feature_engineering(sample_data):
    """Test the feature engineering pipeline end-to-end."""
    # Load data
    loader = DataLoader(filepath=sample_data, file_type="csv")
    df = loader.load()
    assert not df.empty, "Data should not be empty"

    # Load feature columns from settings
    NUMERIC_FEATURES = settings.config.get("numeric_features", [])
    CATEGORICAL_FEATURES = settings.config.get("categorical_features", [])

    # Run pipeline
    X_processed, pipeline = run_feature_engineering(
        df=df,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        use_woe=False
    )

    # Check output shape
    if hasattr(X_processed, "toarray"):
        X_array = X_processed.toarray()
    else:
        X_array = X_processed

    assert X_array.shape[0] == df.shape[0], "Number of rows should match input"

    # Check feature names
    try:
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out(
        )
    except AttributeError:
        feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    assert len(
        feature_names) == X_array.shape[1], "Feature names should match number of columns"
