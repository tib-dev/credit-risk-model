import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DatetimeFeatures(BaseEstimator, TransformerMixin):
    """
    Extract time-based features from transaction timestamps.

    Derived features help capture behavioral and seasonal effects:
    - Hour of day
    - Day of month
    - Month
    - Year
    """

    def __init__(self, datetime_col="TransactionStartTime"):
        """
        Parameters
        ----------
        datetime_col : str
            Column containing transaction timestamps.
        """
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        """
        Fit method (no learned parameters).

        Parameters
        ----------
        X : pandas.DataFrame
        y : ignored

        Returns
        -------
        self
        """
        return self

    def transform(self, X):
        """
        Extract datetime components from the timestamp column.

        Parameters
        ----------
        X : pandas.DataFrame
            Input transaction data.

        Returns
        -------
        pandas.DataFrame
            Dataset with additional datetime-derived features.
        """
        X = X.copy()
        dt = pd.to_datetime(X[self.datetime_col], errors="coerce", utc=True)

        X["transaction_hour"] = dt.dt.hour
        X["transaction_day"] = dt.dt.day
        X["transaction_month"] = dt.dt.month
        X["transaction_year"] = dt.dt.year

        return X
