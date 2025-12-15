from typing import Iterable, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from math import ceil
import datetime as dt


def plot_numerical_distributions(df: pd.DataFrame, columns: list, ncols: int = 2, log_y: bool = True):
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
    log_y : bool
        Whether to use log scale for y-axis.
    """
    nrows = ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(6*ncols, 5*nrows))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f"{col} Distribution")
        if log_y:
            axes[i].set_yscale('log')

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


def plot_daily_transaction_volume(df, time_col="TransactionStartTime", id_col="TransactionId", figsize=(14, 5)):
    """
    Plot daily transaction volume over time.

    Args:
        df (pd.DataFrame): DataFrame containing transactions.
        time_col (str): Timestamp column name.
        id_col (str): Transaction ID column name.
        figsize (tuple): Figure size.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df['Date'] = df[time_col].dt.date

    daily_counts = df.groupby('Date')[id_col].count()

    plt.figure(figsize=figsize)
    daily_counts.plot()
    plt.title("Daily Transaction Volume Over Time")
    plt.xlabel("Date")
    plt.ylabel("Transaction Count")
    plt.tight_layout()
    plt.show()


def plot_transaction_polarity(df: pd.DataFrame):
    """
    Plots the polarity of transaction amounts (Debit, Credit, Zero).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the transaction data with an 'Amount' column.
    """
    # Compute counts for positive, negative, and zero amounts
    positive_count = (df['Amount'] > 0).sum()
    negative_count = (df['Amount'] < 0).sum()
    zero_count = (df['Amount'] == 0).sum()
    total_count = len(df)

    # Print the breakdown
    print(f"Total Transactions: {total_count}")
    print(
        f"Transactions (Amount > 0 / Debit): {positive_count} ({positive_count/total_count:.2%})")
    print(
        f"Transactions (Amount < 0 / Credit): {negative_count} ({negative_count/total_count:.2%})")
    print(
        f"Transactions (Amount = 0): {zero_count} ({zero_count/total_count:.2%})")

    # Create a DataFrame for plotting
    polarity_df = pd.DataFrame({
        'Polarity': ['Debit (>0)', 'Credit (<0)', 'Zero (=0)'],
        'Count': [positive_count, negative_count, zero_count]
    })

    # Plotting the polarity
    plt.figure(figsize=(8, 6))
    sns.barplot(
        x='Polarity',
        y='Count',
        data=polarity_df,
        palette=['#2ca02c', '#d62728', '#1f77b4']
    )
    plt.title('Transaction Polarity Overview', fontsize=16, weight='bold')
    plt.ylabel('Number of Transactions', fontsize=12)
    plt.xlabel('')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotating the bars with the count values
    for index, row in polarity_df.iterrows():
        plt.text(index, row['Count'] + total_count*0.01,
                 f"{row['Count']:,}", ha='center', fontsize=12)

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(
    df: pd.DataFrame,
    categorical_cols: List[str],
    top_n: int = 10,
    ncols: int = 2,
):
    """
    Plot top category frequencies for multiple categorical features
    in a single grid layout.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    categorical_cols : List[str]
        List of categorical column names.
    top_n : int, default=10
        Number of top categories to display per feature.
    ncols : int, default=2
        Number of columns in the subplot grid.
    """

    if not categorical_cols:
        raise ValueError("categorical_cols list is empty.")

    nrows = ceil(len(categorical_cols) / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(7 * ncols, 4 * nrows)
    )
    axes = axes.flatten()

    palette = sns.color_palette("tab10", n_colors=len(categorical_cols))

    for i, col in enumerate(categorical_cols):
        ax = axes[i]

        value_counts = df[col].value_counts().head(top_n)

        sns.barplot(
            x=value_counts.index.astype(str),
            y=value_counts.values,
            ax=ax,
            color=palette[i],
        )

        ax.set_title(
            f"{col} (Top {top_n})",
            fontsize=12,
            weight="bold"
        )
        ax.set_ylabel("Count")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=40)

        # Annotate bars
        for idx, val in enumerate(value_counts.values):
            ax.text(
                idx,
                val,
                f"{val:,}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Categorical Feature Distributions",
        fontsize=16,
        weight="bold",
        y=1.02
    )

    plt.tight_layout()
    plt.show()
