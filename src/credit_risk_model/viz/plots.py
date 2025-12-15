from typing import Iterable, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from math import ceil


def plot_numerical_distributions(df: pd.DataFrame, columns: list, ncols: int = 2):
    """
    Plot histograms with KDE for numerical columns in a grid layout.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing numerical features.
    columns : list
        List of numerical column names to plot.
    ncols : int
        Number of columns in the grid.
    """
    nrows = ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(6*ncols, 4*nrows))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f"{col} Distribution")

    # Turn off unused axes
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def plot_categorical_counts(df: pd.DataFrame, columns: list, ncols: int = 2):
    """
    Plot countplots for categorical columns in a grid layout.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing categorical features.
    columns : list
        List of categorical column names to plot.
    ncols : int
        Number of columns in the grid.
    """
    nrows = ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(6*ncols, 4*nrows))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        sns.countplot(x=col, data=df, ax=axes[i])
        axes[i].set_title(f"{col} Counts")
        axes[i].tick_params(axis='x', rotation=45)

    # Turn off unused axes
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, cols: list = None):
    """
    Plot correlation heatmap for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing numerical features.
    cols : list, optional
        List of numerical columns to compute correlation on. Defaults to all numeric.
    """
    cols = cols or df.select_dtypes(include='number').columns
    corr = df[cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()


def plot_outliers(
    df: pd.DataFrame,
    columns: Iterable[str],
    titles: Iterable[str] | None = None,
    figsize: tuple[int, int] = (15, 5),
) -> None:
    """
    Plot side-by-side boxplots for selected columns.

    Args:
        df: Input DataFrame
        columns: Columns to plot
        titles: Optional titles for each subplot
        figsize: Figure size

    Raises:
        ValueError: If a column does not exist in the DataFrame
    """
    columns = list(columns)
    n_cols = len(columns)

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")

    if titles is None:
        titles = columns
    else:
        titles = list(titles)

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    if n_cols == 1:
        axes = [axes]

    for ax, col, title in zip(axes, columns, titles):
        sns.boxplot(y=df[col], ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(col)

    plt.tight_layout()
    plt.show()
