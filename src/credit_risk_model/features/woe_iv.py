from sklearn.base import BaseEstimator, TransformerMixin
from xverse.transformer import WOE


class WoETransformer(BaseEstimator, TransformerMixin):
    """
    Apply Weight of Evidence (WoE) transformation for scorecard modeling.

    WoE encodes categorical variables into monotonic, interpretable
    numeric representations aligned with default risk.
    """

    def __init__(self, features):
        """
        Parameters
        ----------
        features : list[str]
            Categorical features to be transformed using WoE.
        """
        self.features = features
        self.woe = WOE()

    def fit(self, X, y):
        """
        Fit WoE bins using the target variable.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature matrix.
        y : array-like
            Binary target variable (e.g., is_high_risk).

        Returns
        -------
        self
        """
        self.woe.fit(X[self.features], y)
        return self

    def transform(self, X):
        """
        Replace original categorical features with WoE-transformed values.

        Parameters
        ----------
        X : pandas.DataFrame
            Input dataset.

        Returns
        -------
        pandas.DataFrame
            Dataset with WoE-encoded features.
        """
        X = X.copy()
        X_woe = self.woe.transform(X[self.features])
        X.drop(columns=self.features, inplace=True)
        return X.join(X_woe)
