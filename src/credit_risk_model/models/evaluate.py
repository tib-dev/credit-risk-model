"""
Model evaluation utilities.

Provides standard evaluation metrics for credit risk classification.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_classifier(model, X_test, y_test):
    """
    Evaluate a classification model using standard metrics.

    Parameters
    ----------
    model : sklearn estimator
        Trained classification model.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True labels.

    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]


    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
