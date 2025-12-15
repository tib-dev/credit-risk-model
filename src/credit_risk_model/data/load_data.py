import pandas as pd


class DataLoader:
    """
    DataLoader is responsible for loading raw data from disk
    into pandas DataFrames.

    It does not clean, preprocess, or analyze data.
    Its only responsibility is safe and explicit data loading.
    """

    def __init__(
        self,
        filepath: str,
        file_type: str = "csv",
        **read_kwargs
    ):
        """
        Initialize the data loader.

        Parameters
        ----------
        filepath : str
            Path to the data file.

        file_type : str
            Type of the file to load.
            Supported: 'csv', 'parquet', 'excel', 'json'.

        read_kwargs : dict
            Extra keyword arguments passed directly to
            the pandas reader function.
        """
        self.filepath = filepath
        self.file_type = file_type.lower()
        self.read_kwargs = read_kwargs

    def load(self) -> pd.DataFrame:
        """
        Load the dataset into a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
            Loaded dataset.

        Raises
        ------
        ValueError
            If the file type is not supported.
        """
        if self.file_type == "csv":
            return pd.read_csv(self.filepath, **self.read_kwargs)

        if self.file_type == "parquet":
            return pd.read_parquet(self.filepath, **self.read_kwargs)

        if self.file_type in {"excel", "xlsx"}:
            return pd.read_excel(self.filepath, **self.read_kwargs)

        if self.file_type == "json":
            return pd.read_json(self.filepath, **self.read_kwargs)

        raise ValueError(f"Unsupported file type: {self.file_type}")
