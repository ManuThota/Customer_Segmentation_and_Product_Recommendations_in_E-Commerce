import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# Setup path so that we can import src modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import settings
from src.customer_segmentation import utils as seg_utils
from src.recommendation import similarity as rec_similarity

def generate_eda_plots(df_clean, rfm):
    print("Generating EDA plots...")
    os.makedirs(settings.PLOT_DIR, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    # 1. Transaction Volume by Country
    plt.figure(figsize=(10, 6))
    country_counts = df_clean['Country'].value_counts().head(10)
    sns.barplot(x=country_counts.values, y=country_counts.index, palette="viridis")
    plt.title('Top 10 Countries by Number of Transactions', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Number of Transactions', fontsize=12)
    plt.ylabel('Country', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(settings.PLOT_DIR, 'country_analysis.png'), dpi=300)
    plt.close()
    
    # 2. Top Selling Products
    plt.figure(figsize=(10, 6))
    top_products = df_clean.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=top_products.values, y=top_products.index, palette="plasma")
    plt.title('Top 10 Selling Products by Quantity', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Total Quantity Sold', fontsize=12)
    plt.ylabel('Product Description', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(settings.PLOT_DIR, 'top_products.png'), dpi=300)
    plt.close()
    
    # 3. Top Revenue Generating Products
    plt.figure(figsize=(10, 6))
    top_revenue = df_clean.groupby('Description')['TotalAmount'].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=top_revenue.values, y=top_revenue.index, palette="inferno")
    plt.title('Top 10 Revenue Generating Products', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Total Revenue (£)', fontsize=12)
    plt.ylabel('Product Description', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(settings.PLOT_DIR, 'revenue_analysis.png'), dpi=300)
    plt.close()

def generate_clustering_plots(rfm_scaled, rfm):
    print("Generating Clustering evaluation plots...")
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    inertia = []
    scores = []
    K_range = range(2, 11)
    
    for k in K_range:
        model = KMeans(n_clusters=k, random_state=settings.RANDOM_STATE, n_init=10)
        labels = model.fit_predict(rfm_scaled)
        inertia.append(model.inertia_)
        score = silhouette_score(rfm_scaled, labels)
        scores.append(score)
        print(f"K = {k} | Inertia = {model.inertia_:.2f} | Silhouette Score = {score:.4f}")
        
    # 4. Elbow Curve Plot
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertia, marker='o', linewidth=2, color='#1976D2')
    plt.title('Elbow Method for Optimal Clusters', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Number of Clusters (K)', fontsize=11)
    plt.ylabel('Inertia (WCSS)', fontsize=11)
    plt.xticks(K_range)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(settings.PLOT_DIR, 'elbow_curve.png'), dpi=300)
    plt.close()
    
    # 5. Silhouette Score Plot
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, scores, marker='o', linewidth=2, color='#E53935')
    plt.title('Silhouette Scores for Cluster Selection', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Number of Clusters (K)', fontsize=11)
    plt.ylabel('Silhouette Score', fontsize=11)
    plt.xticks(K_range)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(settings.PLOT_DIR, 'silhouette_score.png'), dpi=300)
    plt.close()
    
    # 6. Cluster Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=rfm,
        x='Frequency',
        y='Monetary',
        hue='Segment',
        palette={'High-Value': '#2E7D32', 'Regular': '#1976D2', 'Occasional': '#F57C00', 'At-Risk': '#D32F2F'},
        alpha=0.7,
        edgecolor='w',
        s=50
    )
    plt.title('Customer Segments: Frequency vs Monetary (Log Scale)', fontsize=14, fontweight='bold', pad=15)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequency (Log Scale)', fontsize=12)
    plt.ylabel('Monetary Value (Log Scale)', fontsize=12)
    plt.legend(title='Customer Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(settings.PLOT_DIR, 'cluster_visualization.png'), dpi=300)
    plt.close()

def main():
    print("Starting Shopper Spectrum Pipeline Execution...")
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(settings.CLEANED_DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(settings.SCALER_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(settings.SAMPLE_INPUTS_PATH), exist_ok=True)
    
    # 1. Load Raw Data
    print(f"Loading raw data from {settings.RAW_DATA_PATH}...")
    if not os.path.exists(settings.RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data file not found at {settings.RAW_DATA_PATH}")
    df_raw = pd.read_csv(settings.RAW_DATA_PATH)
    
    # 2. Clean Data
    print("Cleaning dataset...")
    df_clean = seg_utils.clean_data(df_raw)
    df_clean.to_csv(settings.CLEANED_DATA_PATH, index=False)
    print(f"Cleaned dataset shape: {df_clean.shape}. Saved to {settings.CLEANED_DATA_PATH}")
    
    # 3. Generate EDA Plots
    generate_eda_plots(df_clean, None)
    
    # 4. Generate RFM Table
    print("Computing RFM metrics...")
    rfm = seg_utils.calculate_rfm(df_clean)
    rfm.to_csv(settings.RFM_DATA_PATH)
    print(f"RFM dataset shape: {rfm.shape}. Saved to {settings.RFM_DATA_PATH}")
    
    # 5. Scale RFM features
    print("Scaling features...")
    rfm_scaled, scaler = seg_utils.scale_rfm(rfm)
    
    # Save the scaler
    joblib.dump(scaler, settings.SCALER_PATH)
    print(f"Scaler saved to {settings.SCALER_PATH}")
    
    # 6. Train Clustering Model & Generate Plots
    print("Training KMeans Model & Generating Plots...")
    kmeans = seg_utils.train_kmeans(rfm_scaled, n_clusters=settings.KMEANS_CLUSTERS, random_state=settings.RANDOM_STATE)
    
    # Save KMeans Model
    joblib.dump(kmeans, settings.KMEANS_MODEL_PATH)
    print(f"KMeans model saved to {settings.KMEANS_MODEL_PATH}")
    
    # Assign cluster labels
    rfm['Cluster'] = kmeans.labels_
    rfm['Segment'] = rfm['Cluster'].map(settings.SEGMENT_MAP)
    
    # Save Customer Segments Data
    rfm.to_csv(settings.CUSTOMER_SEGMENTS_PATH)
    print(f"Segmented customer dataset saved to {settings.CUSTOMER_SEGMENTS_PATH}")
    
    # Save Segment Map
    joblib.dump(settings.SEGMENT_MAP, settings.SEGMENT_MAPPING_PATH)
    print(f"Segment mapping saved to {settings.SEGMENT_MAPPING_PATH}")
    
    # Generate clustering analysis plots
    generate_clustering_plots(rfm_scaled, rfm)
    
    # 7. Recommendation Matrix
    print("Building Customer-Product Matrix...")
    customer_product_matrix = rec_similarity.create_customer_product_matrix(df_clean)
    
    # Save customer-product interaction matrix
    joblib.dump(customer_product_matrix, settings.PRODUCT_SIMILARITY_PATH)
    print(f"Customer-Product matrix saved to {settings.PRODUCT_SIMILARITY_PATH}")
    
    print("Computing Cosine Similarity Matrix (this may take a minute)...")
    similarity_df = rec_similarity.calculate_cosine_similarity(customer_product_matrix)
    
    # Save similarity matrix
    joblib.dump(similarity_df, settings.PRODUCT_SIMILARITY_MATRIX_PATH)
    print(f"Product similarity matrix saved to {settings.PRODUCT_SIMILARITY_MATRIX_PATH}")
    
    # Save product names list
    product_names = list(similarity_df.index)
    joblib.dump(product_names, settings.PRODUCT_NAMES_PATH)
    print(f"Product names saved to {settings.PRODUCT_NAMES_PATH}")
    
    # 8. Generate Sample Inputs
    print("Creating sample inputs file...")
    sample_df = rfm.reset_index().sample(n=20, random_state=42)
    sample_df = sample_df[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Segment']]
    sample_df.to_csv(settings.SAMPLE_INPUTS_PATH, index=False)
    print(f"Sample inputs saved to {settings.SAMPLE_INPUTS_PATH}")
    
    print("Shopper Spectrum training pipeline completed successfully!")

if __name__ == "__main__":
    main()
