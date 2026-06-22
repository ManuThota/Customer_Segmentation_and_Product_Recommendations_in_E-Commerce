import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.recommendation.recommender import ProductRecommender

# Page Configuration
st.set_page_config(
    page_title="Product Recommendations - Shopper Spectrum",
    page_icon=None,
    layout="wide"
)

# Custom Style for Recommendation Cards
st.markdown("""
<style>
    .rec-card {
        background-color: #F8FAFC;
        border-left: 5px solid #1E3A8A;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .rec-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0F172A;
    }
    .rec-score {
        font-size: 0.85rem;
        font-weight: 600;
        color: #10B981; /* Emerald green */
        margin-top: 0.25rem;
    }
    .search-hint {
        font-size: 0.85rem;
        color: #64748B;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_PATH = os.path.join(PROJECT_ROOT, "assets", "recommendation_system.png")

# Instantiate Recommender
@st.cache_resource
def get_recommender():
    return ProductRecommender()

try:
    recommender = get_recommender()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure the training pipeline has been completed successfully.")
    recommender = None

# Header Image / Title
if os.path.exists(IMAGE_PATH):
    st.image(IMAGE_PATH, use_column_width=True)
else:
    st.title("Product Recommendation Module")
    st.markdown("Discover item associations and similar products using collaborative filtering.")

st.markdown("---")

# Main Interface Layout
col_search, col_results = st.columns([1, 2])

with col_search:
    st.subheader("Product Query Search")
    
    if recommender is not None:
        # Product autocomplete helper - since there are 3800+ names, let's allow typing a keyword to filter the list
        search_keyword = st.text_input("Type a keyword to filter product list", value="BAG", help="e.g. BAG, HEART, BOX, SIGN")
        
        # Filter product list based on keyword
        filtered_names = [name for name in recommender.product_names if search_keyword.upper() in name.upper()]
        
        if not filtered_names:
            st.warning("No products match the keyword filter. Showing first 100 products instead.")
            filtered_names = recommender.product_names[:100]
            
        selected_product = st.selectbox("Select Target Product Name", sorted(filtered_names), help="Select the exact product to query similar items for.")
        
        num_recommendations = st.slider("Number of Recommendations", min_value=3, max_value=10, value=5)
        
        get_rec_btn = st.button("Get Recommendations", use_container_width=True)
    else:
        st.error("Recommender model not loaded.")
        get_rec_btn = False

with col_results:
    st.subheader("Top Recommendations")
    
    if get_rec_btn and recommender is not None:
        # Run recommendation
        res = recommender.recommend(selected_product, n=num_recommendations)
        
        if res is not None:
            actual_product = res['selected_product']
            recs = res['recommendations']
            
            st.success(f"Recommendations generated for: **{actual_product}**")
            
            # Display matching cards
            for i, rec in enumerate(recs):
                pct_match = rec['similarity'] * 100
                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-title">{i+1}. {rec['product']}</div>
                    <div class="rec-score">Similarity Match: {pct_match:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("---")
            
            # Dynamic Correlation / Similarity Heatmap
            st.subheader("Recommended Product Similarity Matrix")
            st.write("Visualizing the correlation levels between the queried item and the recommended items:")
            
            try:
                # Include query item and top N in the submatrix
                sub_products = [actual_product] + [r['product'] for r in recs]
                sub_matrix = recommender.similarity_df.loc[sub_products, sub_products]
                
                # Plot Heatmap
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.heatmap(
                    sub_matrix,
                    annot=True,
                    cmap='Blues',
                    fmt='.2f',
                    linewidths=0.5,
                    cbar=True,
                    ax=ax
                )
                plt.title('Similarity Level Mapping', fontsize=12, fontweight='bold', pad=12)
                plt.xticks(rotation=45, ha='right', fontsize=9)
                plt.yticks(rotation=0, fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            except Exception as e_plot:
                st.write(f"Could not generate heatmap: {e_plot}")
                
        else:
            st.error("No similar products found.")
    else:
        st.info("Select a product on the left pane and click 'Get Recommendations' to see similar items.")
