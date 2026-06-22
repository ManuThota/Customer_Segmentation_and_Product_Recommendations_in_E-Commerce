import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from src.config.settings import SEGMENT_MAP

def clean_data(df):
    """
    Cleans raw Online Retail transaction data.
    - Removes rows with missing CustomerID
    - Excludes cancelled invoices (InvoiceNo starting with 'C')
    - Removes negative/zero quantities and unit prices
    - Removes duplicate records
    - Converts InvoiceDate to datetime
    - Creates TotalAmount feature
    """
    df_clean = df.copy()
    
    # Remove missing CustomerID
    df_clean = df_clean.dropna(subset=['CustomerID'])
    
    # Exclude cancelled invoices
    df_clean = df_clean[~df_clean['InvoiceNo'].str.startswith('C', na=False)]
    
    # Exclude zero or negative prices and quantities
    df_clean = df_clean[df_clean['UnitPrice'] > 0]
    df_clean = df_clean[df_clean['Quantity'] > 0]
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Convert InvoiceDate to Datetime
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    
    # Create TotalAmount
    df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']
    
    return df_clean

def calculate_rfm(df_clean):
    """
    Calculates Recency, Frequency, and Monetary metrics for each customer.
    - Recency: Days since customer's last purchase relative to reference date.
    - Frequency: Number of unique invoices per customer.
    - Monetary: Total spend per customer.
    """
    latest_date = df_clean['InvoiceDate'].max()
    reference_date = latest_date + pd.Timedelta(days=1)
    
    rfm = df_clean.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    })
    
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm

def scale_rfm(rfm_df, scaler=None):
    """
    Transforms RFM metrics:
    - Log transforms Frequency and Monetary.
    - Scales features to zero mean and unit variance.
    If scaler is None, fits a new StandardScaler and returns (scaled_df, scaler).
    If scaler is provided, transforms the metrics and returns scaled_df.
    """
    rfm_log = rfm_df.copy()
    
    # Log transform Frequency and Monetary (Recency is NOT log transformed as per notebook)
    rfm_log['Frequency'] = np.log1p(rfm_log['Frequency'])
    rfm_log['Monetary'] = np.log1p(rfm_log['Monetary'])
    
    if scaler is None:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(rfm_log)
        scaled_df = pd.DataFrame(scaled_values, columns=rfm_df.columns, index=rfm_df.index)
        return scaled_df, scaler
    else:
        scaled_values = scaler.transform(rfm_log)
        scaled_df = pd.DataFrame(scaled_values, columns=rfm_df.columns, index=rfm_df.index)
        return scaled_df

def train_kmeans(scaled_rfm, n_clusters=4, random_state=42):
    """
    Fits K-Means clustering algorithm on scaled RFM dataset.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(scaled_rfm)
    return kmeans
