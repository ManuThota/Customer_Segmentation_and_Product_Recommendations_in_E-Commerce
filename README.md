# 🛒 Shopper Spectrum: Customer Segmentation and Product Recommendations in E-Commerce

Shopper Spectrum is an end-to-end data science project designed to extract actionable customer behavior insights and provide personalized product recommendations using e-commerce transaction logs.

The system utilizes Recency, Frequency, and Monetary (RFM) analysis combined with unsupervised K-Means clustering to group customers. Additionally, it applies item-based collaborative filtering (Cosine Similarity) to suggest similar products based on historical purchase patterns.

---

## Key Features
1. **Advanced Customer Preprocessing & Cleaning**: 
   - Excludes cancelled orders and returned items.
   - Cleans records with null Customer IDs and invalid quantity or price metrics.
   - Converts dates and scales transaction features.
2. **Behavioral Customer Segmentation**:
   - Calculates RFM metrics dynamically relative to a reference date.
   - Normalizes data skew using log transforms.
   - Trains K-Means clustering (K=4) to classify profiles into: **High-Value**, **Regular**, **Occasional**, and **At-Risk** customers.
3. **Collaborative Filtering Recommender**:
   - Computes product similarity using Cosine Similarity on customer-item interaction matrix.
   - Recommends 5 matching products based on text key queries.
4. **Interactive Dashboard & Business Report**:
   - Renders statistics, EDA galleries, interactive inputs, and batch uploads in Streamlit.
   - Compiles a professional PDF summary report containing all analytics.

---

## Project Structure

```
Shopper_Spectrum/
│
├── app.py                      # Streamlit homepage and dashboard metrics
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore file
│
├── assets/                     # Premium app image assets
│   ├── customer_segmentation.png
│   ├── recommendation_system.png
│   └── banner.png
│
├── data/
│   ├── raw/
│   │   └── OnlineRetail.csv    # Raw dataset (copied from workspace root)
│   │
│   ├── processed/
│   │   ├── cleaned_online_retail.csv
│   │   ├── rfm_dataset.csv
│   │   └── customer_segments.csv
│   │
│   └── sample/
│       └── sample_inputs.csv   # 20 random customer samples for testing
│
├── models/                     # Saved models and encoders
│   ├── scaler.pkl
│   ├── kmeans_model.pkl
│   ├── segment_mapping.pkl
│   ├── product_similarity.pkl
│   ├── product_similarity_matrix.pkl
│   └── product_names.pkl
│
├── notebooks/                  # Project Jupyter notebooks
│   └── Shopper_Spectrum.ipynb
│
├── src/                        # Modular source package
│   ├── __init__.py
│   │
│   ├── customer_segmentation/
│   │   ├── __init__.py
│   │   ├── predictor.py        # Customer Segment prediction class
│   │   └── utils.py            # Preprocessing and cluster fitting functions
│   │
│   ├── recommendation/
│   │   ├── __init__.py
│   │   ├── recommender.py      # Product recommendation engine class
│   │   └── similarity.py       # Matrix pivot and cosine similarity builders
│   │
│   └── config/
│       ├── __init__.py
│       └── settings.py         # Global settings and path definitions
│
├── pages/                      # Multi-page Streamlit views
│   ├── 1_Customer_Segmentation.py
│   └── 2_Product_Recommendation.py
│
└── outputs/                    # Output reports and visualization plots
    ├── plots/
    │   ├── country_analysis.png
    │   ├── top_products.png
    │   ├── revenue_analysis.png
    │   ├── elbow_curve.png
    │   ├── silhouette_score.png
    │   └── cluster_visualization.png
    │
    └── reports/
        └── project_summary.pdf # PDF Summary report
```

---

## Setup & Execution

### 1. Prerequisites
Verify you have python 3.8+ installed:
```bash
python --version
```

### 2. Install Dependencies
Run the following command to install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Execution Pipeline
Run the preprocessing and training pipeline to generate the models, plots, sample data, and PDF report:
```bash
# From the Shopper_Spectrum directory:
python run_pipeline.py
python generate_report.py
```

### 4. Run the Streamlit Application
Launch the multi-page web application locally:
```bash
streamlit run app.py
```

---

## Business Personas & Strategies
- **High-Value Spenders**: Target with exclusive VIP campaigns, premium early product access, and dedicated account incentives.
- **Regular Customers**: Push cross-selling, loyalty milestone rewards, and persistent newsletters.
- **Occasional Shoppers**: Re-engage with seasonal sales, flash discounts, and tailored recommendation items.
- **At-Risk Customers**: Launch reactivating win-back coupon offers and distribute feedback surveys.
