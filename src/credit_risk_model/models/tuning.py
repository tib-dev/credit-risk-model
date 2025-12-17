"""
Hyperparameter tuning utilities.
"""

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def tune_model(
    model,
    tuning_cfg: dict,
    X_train,
    y_train,
    optimize_metric: str,
):
    """
    Tune model using GridSearch or RandomizedSearch based on config.
    """

    search_type = tuning_cfg["search_type"]
    params = tuning_cfg["params"]
    cv = tuning_cfg.get("cv", 3)

    if search_type == "grid":
        search = GridSearchCV(
            model,
            param_grid=params,
            scoring=optimize_metric,
            cv=cv,
            n_jobs=-1,
        )
    else:
        search = RandomizedSearchCV(
            model,
            param_distributions=params,
            n_iter=20,
            scoring=optimize_metric,
            cv=cv,
            random_state=42,
            n_jobs=-1,
        )

    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_
