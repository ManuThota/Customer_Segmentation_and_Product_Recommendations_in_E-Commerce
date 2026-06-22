import joblib
import pandas as pd
import os
from src.config.settings import PRODUCT_SIMILARITY_MATRIX_PATH, PRODUCT_NAMES_PATH

class ProductRecommender:
    def __init__(self, matrix_path=PRODUCT_SIMILARITY_MATRIX_PATH, names_path=PRODUCT_NAMES_PATH):
        """
        Loads the saved similarity matrix and product names.
        """
        if not os.path.exists(matrix_path) or not os.path.exists(names_path):
            raise FileNotFoundError("Recommendation models are missing. Please run the training pipeline first.")
            
        self.similarity_df = joblib.load(matrix_path)
        self.product_names = joblib.load(names_path)
        
    def recommend(self, product_name, n=5):
        """
        Recommends top N products similar to the query.
        Handles substring matching and case insensitivity.
        """
        query = product_name.upper().strip()
        
        # Find exact or partial matching products
        matching_products = [
            prod for prod in self.similarity_df.index
            if query in prod.upper()
        ]
        
        if not matching_products:
            return None
            
        # Select the first match
        selected_product = matching_products[0]
        
        # Get recommendations (sorting descending, drop the queried product)
        recommendations = self.similarity_df[selected_product].sort_values(ascending=False)
        recommendations = recommendations.drop(labels=[selected_product], errors='ignore')
        
        top_n = recommendations.head(n)
        
        results = []
        for prod, score in top_n.items():
            results.append({
                "product": prod,
                "similarity": float(score)
            })
            
        return {
            "selected_product": selected_product,
            "recommendations": results
        }
