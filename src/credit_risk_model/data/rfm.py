# rfm.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_rfm(df: pd.DataFrame, customer_id_col='CustomerId',
                  transaction_date_col='TransactionStartTime',
                  amount_col='Value', snapshot_date=None) -> pd.DataFrame:
    """
    Calculate RFM metrics for each customer.
    
    Parameters:
        df (pd.DataFrame): Transaction data
        customer_id_col (str): Column name for customer ID
        transaction_date_col (str): Column name for transaction date
        amount_col (str): Column name for transaction amount
        snapshot_date (pd.Timestamp, optional): Reference date for recency. Defaults to max date in df.
        
    Returns:
        pd.DataFrame: RFM table with columns [CustomerId, Recency, Frequency, Monetary]
    """
    logging.info("Calculating RFM metrics...")

    df[transaction_date_col] = pd.to_datetime(df[transaction_date_col])

    if snapshot_date is None:
        snapshot_date = df[transaction_date_col].max() + pd.Timedelta(days=1)

    rfm = df.groupby(customer_id_col).agg(
        Recency=(transaction_date_col, lambda x: (snapshot_date - x.max()).days),
        Frequency=(transaction_date_col, 'count'),
        Monetary=(amount_col, 'sum')
    ).reset_index()

    return rfm



def cluster_rfm(rfm_df: pd.DataFrame, n_clusters=3, random_state=42) -> pd.DataFrame:
    """
    Cluster customers based on RFM values using K-Means.
    
    Parameters:
        rfm_df (pd.DataFrame): DataFrame with RFM metrics
        n_clusters (int): Number of clusters
        random_state (int): Random state for reproducibility
        
    Returns:
        pd.DataFrame: RFM table with an additional 'Cluster' column
    """
    logging.info("Clustering customers using K-Means...")

    features = ['Recency', 'Frequency', 'Monetary']
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

    logging.info("Clustering completed successfully.")
    return rfm_df


def assign_high_risk_label(rfm_clustered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign high-risk label to customers based on cluster characteristics.
    
    Parameters:
        rfm_clustered_df (pd.DataFrame): RFM DataFrame with clusters
        
    Returns:
        pd.DataFrame: DataFrame with 'is_high_risk' column added
    """
    logging.info("Assigning high-risk label based on cluster RFM scores...")

    # Calculate average RFM per cluster
    cluster_summary = rfm_clustered_df.groupby(
        'Cluster')[['Recency', 'Frequency', 'Monetary']].mean()

    # Heuristic: High-risk = highest recency, lowest frequency and monetary
    cluster_summary['RiskScore'] = cluster_summary['Recency'].rank(ascending=False) + \
        cluster_summary['Frequency'].rank(ascending=True) + \
        cluster_summary['Monetary'].rank(ascending=True)

    high_risk_cluster = cluster_summary['RiskScore'].idxmax()

    rfm_clustered_df['is_high_risk'] = (
        rfm_clustered_df['Cluster'] == high_risk_cluster).astype(int)

    logging.info(f"High-risk cluster identified: {high_risk_cluster}")
    return rfm_clustered_df


def create_rfm_target(df: pd.DataFrame, customer_id_col='CustomerId', transaction_date_col='TransactionStartTime',
                      amount_col='Value', snapshot_date=None, n_clusters=3, random_state=42) -> pd.DataFrame:
    """
    Full RFM -> cluster -> high-risk label pipeline.
    
    Returns:
        pd.DataFrame: RFM DataFrame with 'is_high_risk'
    """
    rfm_df = calculate_rfm(df, customer_id_col,
                           transaction_date_col, amount_col, snapshot_date)
    rfm_clustered_df = cluster_rfm(rfm_df, n_clusters, random_state)
    rfm_labeled_df = assign_high_risk_label(rfm_clustered_df)
    return rfm_labeled_df
