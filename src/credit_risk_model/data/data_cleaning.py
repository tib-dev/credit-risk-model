import pandas as pd
import numpy as np
from typing import Optional, List


class DataCleaning:
    """
    DataCleaning performs deterministic, rule-based cleaning on raw datasets.

    This module is intentionally separate from preprocessing pipelines.
    It handles obvious data quality issues before feature engineering
    and model-ready preprocessing.
    """

    def __init__(
        self,
        drop_duplicates: bool = True,
        strip_strings: bool = True,
        empty_string_as_nan: bool = True,
        datetime_columns: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None,
    ):
        """
        Initialize data cleaning configuration.

        Parameters
        ----------
        drop_duplicates : bool
            Remove duplicate rows.
        strip_strings : bool
            Strip leading/trailing spaces from string values.
        empty_string_as_nan : bool
            Convert empty strings to NaN.
        datetime_columns : list of str, optional
            Columns to convert to datetime format.
        numeric_columns : list of str, optional
            Columns to ensure numeric type, fill NaN with 0.
        """
        self.drop_duplicates = drop_duplicates
        self.strip_strings = strip_strings
        self.empty_string_as_nan = empty_string_as_nan
        self.datetime_columns = datetime_columns or []
        self.numeric_columns = numeric_columns or []

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data cleaning steps to the input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Raw input dataset.

        Returns
        -------
        pd.DataFrame
            Cleaned dataset.
        """
        df = df.copy()


        if self.strip_strings:
            df = self._strip_string_values(df)

        if self.empty_string_as_nan:
            df.replace("", np.nan, inplace=True)

        # Convert specified columns to datetime
        for col in self.datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Convert numeric columns
        for col in self.numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        if self.drop_duplicates:
            df.drop_duplicates(inplace=True)

        return df

    def _strip_string_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strip whitespace from string columns only.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        string_cols = df.select_dtypes(include="object").columns
        for col in string_cols:
            df[col] = df[col].str.strip()
        return df
