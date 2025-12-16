from sklearn.base import BaseEstimator, TransformerMixin


class CustomerAggregates(BaseEstimator, TransformerMixin):
    """
    Generate customer-level aggregate transaction features.

    This transformer computes summary statistics per customer, which
    capture overall spending behavior and transaction variability.

    Aggregates created:
    - total_transaction_amount: Sum of transaction values
    - avg_transaction_amount: Mean transaction value
    - transaction_count: Number of transactions
    - std_transaction_amount: Transaction value variability
    """

    def __init__(self, customer_col="CustomerId", value_col="Value"):
        """
        Parameters
        ----------
        customer_col : str
            Column identifying unique customers.
        value_col : str
            Absolute transaction value column used for aggregation.
        """
        self.customer_col = customer_col
        self.value_col = value_col

    def fit(self, X, y=None):
        """
        Fit method (no learned parameters).

        Parameters
        ----------
        X : pandas.DataFrame
            Input transaction data.
        y : ignored

        Returns
        -------
        self
        """
        return self

    def transform(self, X):
        """
        Compute and merge aggregate features back into the dataset.

        Parameters
        ----------
        X : pandas.DataFrame
            Input transaction data.

        Returns
        -------
        pandas.DataFrame
            Dataset enriched with customer-level aggregate features.
        """
        X = X.copy()

        agg = (
            X.groupby(self.customer_col)[self.value_col]
            .agg(
                total_transaction_amount="sum",
                avg_transaction_amount="mean",
                transaction_count="count",
                std_transaction_amount="std",
            )
            .reset_index()
        )

        return X.merge(agg, on=self.customer_col, how="left")
