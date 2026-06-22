import streamlit as st
import os
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Shopper Spectrum - E-Commerce Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title & Style
st.markdown("""
<style>
    .main-title {
        font-family: 'Outfit', sans-serif;
        color: #0F172A;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #64748B;
        font-size: 1.25rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# Helper to load paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BANNER_PATH = os.path.join(PROJECT_ROOT, "assets", "banner.png")
PDF_PATH = os.path.join(PROJECT_ROOT, "outputs", "reports", "project_summary.pdf")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")

# Header Section with Banner
if os.path.exists(BANNER_PATH):
    st.image(BANNER_PATH, use_column_width=True)
else:
    st.markdown('<div class="main-title">Shopper Spectrum</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Customer Segmentation & Product Recommendations in E-Commerce</div>', unsafe_allow_html=True)

st.markdown("---")

# Main Metrics Dashboard
st.subheader("Database Analytics Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">392,692</div>
        <div class="metric-label">Valid Transactions</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">4,338</div>
        <div class="metric-label">Segmented Customers</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">3,877</div>
        <div class="metric-label">Unique Products</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">38</div>
        <div class="metric-label">Countries Covered</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("Spacer") # Visual spacer
st.write("")

# Two-column layout for overview and PDF download
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Solution Overview")
    st.write(
        "**Shopper Spectrum** is an end-to-end data science application that helps e-commerce "
        "businesses optimize their marketing and product presentation through customer intelligence."
    )
    
    st.write("The application comprises two core machine learning pipelines:")
    
    st.markdown("""
    1. **Customer Segmentation Engine (RFM + K-Means)**:
       - Engineers Recency, Frequency, and Monetary (RFM) characteristics per customer.
       - Applies logarithmic transforms to resolve feature skewness.
       - Clusters customers into 4 behavioral segments: **High-Value**, **Regular**, **Occasional**, and **At-Risk**.
       
    2. **Product Recommender Engine (Collaborative Filtering)**:
       - Uses item-based collaborative filtering based on historical purchases.
       - Measures similarity between items using Cosine Similarity metrics.
       - Recommends the top 5 most similar products based on user search.
    """)
    
    st.info("Use the sidebar menu to navigate to Customer Segmentation or Product Recommendation tools.")

with col_right:
    st.subheader("Executive Summary Report")
    st.write(
        "A formal Business and Technical summary has been generated as a PDF report, including data preprocessing details, K-Means cluster evaluations, and segment retention strategies."
    )
    
    if os.path.exists(PDF_PATH):
        with open(PDF_PATH, "rb") as f:
            pdf_data = f.read()
            st.download_button(
                label="Download Project Summary PDF",
                data=pdf_data,
                file_name="Shopper_Spectrum_Summary_Report.pdf",
                mime="application/pdf"
            )
    else:
        st.warning("Project summary report PDF not found. Please run the pipeline script to compile the PDF.")

st.markdown("---")

# Visual Gallery of Generated Plots
st.subheader("Exploratory Insights Gallery")
gallery_tabs = st.tabs(["Geographic Volatility", "Revenue Products", "Top Products"])

with gallery_tabs[0]:
    plot_path = os.path.join(PLOTS_DIR, "country_analysis.png")
    if os.path.exists(plot_path):
        st.image(plot_path, caption="Top Countries by Transaction Count", use_column_width=True)
    else:
        st.info("Country analysis plot not found.")

with gallery_tabs[1]:
    plot_path = os.path.join(PLOTS_DIR, "revenue_analysis.png")
    if os.path.exists(plot_path):
        st.image(plot_path, caption="Top 10 Products by Total Revenue Generated", use_column_width=True)
    else:
        st.info("Revenue analysis plot not found.")

with gallery_tabs[2]:
    plot_path = os.path.join(PLOTS_DIR, "top_products.png")
    if os.path.exists(plot_path):
        st.image(plot_path, caption="Top 10 Products by Quantity Sold", use_column_width=True)
    else:
        st.info("Top products plot not found.")
