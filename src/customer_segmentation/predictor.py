import joblib
import pandas as pd
import numpy as np
import os
from src.config.settings import SCALER_PATH, KMEANS_MODEL_PATH, SEGMENT_MAPPING_PATH

class CustomerSegmentPredictor:
    def __init__(self, scaler_path=SCALER_PATH, kmeans_path=KMEANS_MODEL_PATH, mapping_path=SEGMENT_MAPPING_PATH):
        """
        Loads the trained Standard Scaler, K-Means Clustering model, and segment mapping dict.
        """
        if not os.path.exists(scaler_path) or not os.path.exists(kmeans_path) or not os.path.exists(mapping_path):
            raise FileNotFoundError("Clustering models/mappings are missing. Please run the training pipeline first.")
            
        self.scaler = joblib.load(scaler_path)
        self.kmeans = joblib.load(kmeans_path)
        self.segment_mapping = joblib.load(mapping_path)
        
    def predict(self, recency, frequency, monetary):
        """
        Predicts the customer segment given raw RFM input values.
        - recency: Days since last purchase (int or float)
        - frequency: Number of transactions (int or float)
        - monetary: Total spend (float)
        """
        # Create input DataFrame
        input_data = pd.DataFrame([{
            'Recency': float(recency),
            'Frequency': float(frequency),
            'Monetary': float(monetary)
        }])
        
        # Apply the log1p transform (Frequency and Monetary ONLY)
        input_data['Frequency'] = np.log1p(input_data['Frequency'])
        input_data['Monetary'] = np.log1p(input_data['Monetary'])
        
        # Standardize using the loaded scaler
        scaled_values = self.scaler.transform(input_data)
        scaled_data = pd.DataFrame(scaled_values, columns=input_data.columns)
        
        # Predict the cluster ID
        cluster_id = self.kmeans.predict(scaled_data)[0]
        
        # Map the cluster ID to its label
        segment_label = self.segment_mapping.get(cluster_id, "Unknown")
        
        return {
            "cluster_id": int(cluster_id),
            "segment": segment_label
        }
