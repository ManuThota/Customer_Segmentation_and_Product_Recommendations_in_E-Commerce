import os

# Project root directory
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Data paths
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "OnlineRetail.csv")
CLEANED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_online_retail.csv")
RFM_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "rfm_dataset.csv")
CUSTOMER_SEGMENTS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "customer_segments.csv")
SAMPLE_INPUTS_PATH = os.path.join(PROJECT_ROOT, "data", "sample", "sample_inputs.csv")

# Model paths
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
KMEANS_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "kmeans_model.pkl")
SEGMENT_MAPPING_PATH = os.path.join(PROJECT_ROOT, "models", "segment_mapping.pkl")
PRODUCT_SIMILARITY_PATH = os.path.join(PROJECT_ROOT, "models", "product_similarity.pkl")
PRODUCT_SIMILARITY_MATRIX_PATH = os.path.join(PROJECT_ROOT, "models", "product_similarity_matrix.pkl")
PRODUCT_NAMES_PATH = os.path.join(PROJECT_ROOT, "models", "product_names.pkl")

# Outputs paths
PLOT_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")
REPORT_PATH = os.path.join(PROJECT_ROOT, "outputs", "reports", "project_summary.pdf")

# Clustering Configs
KMEANS_CLUSTERS = 4
RANDOM_STATE = 42

# Segment Labels Mapping
SEGMENT_MAP = {
    0: 'Occasional',
    1: 'At-Risk',
    2: 'High-Value',
    3: 'Regular'
}

# Segment Profiles for app descriptions
SEGMENT_DESCRIPTIONS = {
    'High-Value': {
        'characteristics': 'Recent purchases, high purchase frequency, and premium spending behavior.',
        'action': 'VIP customer service, early access to new collections, loyalty programs, and personalized outreach.',
        'color': '#2E7D32'  # Green
    },
    'Regular': {
        'characteristics': 'Consistent, steady purchases with moderate spending. Core revenue driver.',
        'action': 'Personalized product recommendations, volume-based discounts, cross-selling, and newsletter updates.',
        'color': '#1976D2'  # Blue
    },
    'Occasional': {
        'characteristics': 'Low purchase frequency, lower spending, but recent engagements are possible.',
        'action': 'Flash sales, limited-time discount coupons, re-engagement campaigns, and product recommendations.',
        'color': '#F57C00'  # Orange
    },
    'At-Risk': {
        'characteristics': 'Lapsed customers who have not made a purchase in a long time, with low frequency and spend.',
        'action': 'Win-back campaigns, feedback surveys, deep discount incentives, and direct customer outreach.',
        'color': '#D32F2F'  # Red
    }
}
