import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.customer_segmentation.predictor import CustomerSegmentPredictor
from src.config.settings import SEGMENT_DESCRIPTIONS, PLOT_DIR, SAMPLE_INPUTS_PATH, CUSTOMER_SEGMENTS_PATH

# Page Configuration
st.set_page_config(
    page_title="Customer Segmentation - Shopper Spectrum",
    page_icon=None,
    layout="wide"
)

# Custom Style for Segment Prediction Cards
st.markdown("""
<style>
    .prediction-header {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .segment-card {
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        font-family: 'Inter', sans-serif;
    }
    .segment-title {
        font-size: 2.25rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
    }
    .segment-info {
        font-size: 1rem;
        line-height: 1.5;
        margin-bottom: 1rem;
        opacity: 0.9;
    }
    .action-header {
        font-weight: 700;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
        opacity: 0.8;
    }
    .action-body {
        font-size: 1rem;
        font-weight: 500;
        background: rgba(255,255,255,0.15);
        padding: 0.75rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_PATH = os.path.join(PROJECT_ROOT, "assets", "customer_segmentation.png")

# Instantiate Predictor
@st.cache_resource
def get_predictor():
    return CustomerSegmentPredictor()

try:
    predictor = get_predictor()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure the training pipeline has been completed successfully.")
    predictor = None

# Header Image / Title
if os.path.exists(IMAGE_PATH):
    st.image(IMAGE_PATH, use_column_width=True)
else:
    st.title("Customer Segmentation Module")
    st.markdown("Predict customer behavior segments and discover retention strategies.")

st.markdown("---")

# Main Page Layout: Left panel for inputs, Right panel for outputs
col_inputs, col_outputs = st.columns([1, 2])

with col_inputs:
    st.subheader("Input Customer Metrics")
    
    # Selection Mode
    input_mode = st.radio("Choose Input Mode", ["Manual Input", "Select Sample Customer", "Batch CSV Upload"])
    
    recency = 30
    frequency = 5
    monetary = 500.0
    
    if input_mode == "Manual Input":
        recency = st.number_input("Recency (Days since last purchase)", min_value=1, max_value=365, value=30, step=1, help="Number of days since the customer's last order.")
        frequency = st.number_input("Frequency (Total Transactions)", min_value=1, max_value=500, value=5, step=1, help="Total unique invoices placed by the customer.")
        monetary = st.number_input("Monetary Value (Total Expenditure £)", min_value=0.1, max_value=500000.0, value=500.0, step=10.0, help="Total aggregate amount spent by the customer.")
        
    elif input_mode == "Select Sample Customer":
        if os.path.exists(SAMPLE_INPUTS_PATH):
            sample_df = pd.read_csv(SAMPLE_INPUTS_PATH)
            selected_id = st.selectbox("Select Customer ID", sample_df['CustomerID'].unique())
            row = sample_df[sample_df['CustomerID'] == selected_id].iloc[0]
            
            # Preset values
            recency = int(row['Recency'])
            frequency = int(row['Frequency'])
            monetary = float(row['Monetary'])
            
            st.info(f"CustomerID {selected_id} details loaded.\n\nActual Segment: **{row['Segment']}**")
            st.metric("Recency", f"{recency} days")
            st.metric("Frequency", f"{frequency} purchases")
            st.metric("Monetary", f"£{monetary:,.2f}")
        else:
            st.warning("Sample inputs file not found. Run training pipeline first.")
            
    else: # Batch CSV Upload
        uploaded_file = st.file_uploader("Upload CSV File (must contain Recency, Frequency, Monetary)", type=['csv'])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")

    predict_btn = st.button("Predict Segment", use_container_width=True)

with col_outputs:
    st.subheader("Classification Output")
    
    if predict_btn and input_mode != "Batch CSV Upload":
        if predictor is not None:
            res = predictor.predict(recency, frequency, monetary)
            segment = res['segment']
            desc = SEGMENT_DESCRIPTIONS.get(segment, {"characteristics": "No description available.", "action": "No actions recommended.", "color": "#64748B"})
            
            # Output Card
            st.markdown(f"""
            <div class="segment-card" style="background-color: {desc['color']};">
                <div class="segment-title">{segment} Customer</div>
                <div class="segment-info"><b>Characteristics:</b> {desc['characteristics']}</div>
                <div class="action-header">Recommended Business Retention Action</div>
                <div class="action-body">{desc['action']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display Gauge / Context
            st.subheader("Input Parameters Context")
            st.write(f"Customer profile details: **Recency** of {recency} days, **Frequency** of {frequency} transactions, and a total spend (**Monetary**) of £{monetary:,.2f}.")
        else:
            st.error("Predictor model not initialized.")
            
    elif predict_btn and input_mode == "Batch CSV Upload":
        if uploaded_file is not None and predictor is not None:
            try:
                # Add check for required columns
                required_cols = {'Recency', 'Frequency', 'Monetary'}
                if not required_cols.issubset(batch_df.columns):
                    st.error("Uploaded CSV must contain 'Recency', 'Frequency', and 'Monetary' columns.")
                else:
                    # Run predictions
                    predictions = []
                    for idx, row in batch_df.iterrows():
                        pred = predictor.predict(row['Recency'], row['Frequency'], row['Monetary'])
                        predictions.append(pred['segment'])
                    batch_df['Predicted_Segment'] = predictions
                    
                    st.dataframe(batch_df)
                    
                    # Provide Download link
                    csv_data = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Classified CSV",
                        data=csv_data,
                        file_name="segmented_customers_batch.csv",
                        mime="text/csv"
                    )
            except Exception as ex:
                st.error(f"Error executing batch classification: {ex}")
        else:
            st.info("Please upload a CSV file and click 'Predict Segment'.")
            
    else:
        st.info("Provide customer metrics on the left pane and click 'Predict Segment' to classify customer behavior profiles.")

st.markdown("---")

# Analytics Tabs (Profiles, Model Evaluation, Segments distribution)
st.subheader("Customer Analytics & Model Evaluations")
tab_profiles, tab_eval, tab_dist = st.tabs(["Segment Personas", "Model Performance", "Dataset Distribution"])

with tab_profiles:
    st.markdown("### Profile Characteristics of Segments")
    st.write(
        "Based on the average RFM characteristics computed during model training, the segments are defined as follows:"
    )
    
    # Load segment profiles if exists
    if os.path.exists(CUSTOMER_SEGMENTS_PATH):
        cust_seg_df = pd.read_csv(CUSTOMER_SEGMENTS_PATH)
        profile_df = cust_seg_df.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean()
        profile_df.columns = ['Avg Recency (Days)', 'Avg Frequency (Transactions)', 'Avg Monetary (£)']
        profile_df = profile_df.round(2)
        st.dataframe(profile_df, use_container_width=True)
    else:
        st.info("Segment profile dataset not found. Run training pipeline first.")

with tab_eval:
    st.markdown("### KMeans Clustering Performance Metrics")
    col_elbow, col_sil = st.columns(2)
    
    with col_elbow:
        elbow_path = os.path.join(PLOT_DIR, "elbow_curve.png")
        if os.path.exists(elbow_path):
            st.image(elbow_path, caption="Elbow WCSS Curve", use_column_width=True)
        else:
            st.info("Elbow curve image missing.")
            
    with col_sil:
        sil_path = os.path.join(PLOT_DIR, "silhouette_score.png")
        if os.path.exists(sil_path):
            st.image(sil_path, caption="Silhouette Score Curve", use_column_width=True)
        else:
            st.info("Silhouette score image missing.")
            
    st.markdown("""
    **Model Selection Analysis**:
    - The Elbow WCSS curve indicates a distinct reduction rate shift (inflection point/elbow) around **K=3** and **K=4** clusters.
    - Although the highest overall Silhouette Score occurred at **K=3** (0.4157), the score for **K=4** remains very strong (0.3795).
    - To support targeted marketing strategies, the business segment mapping utilizes **K=4** distinct customer profiles (High-Value, Regular, Occasional, and At-Risk).
    """)

with tab_dist:
    st.markdown("### Segment Distribution Breakdown")
    col_dist_plot, col_dist_table = st.columns([2, 1])
    
    with col_dist_plot:
        dist_plot_path = os.path.join(PLOT_DIR, "cluster_visualization.png")
        if os.path.exists(dist_plot_path):
            st.image(dist_plot_path, caption="Customer Clusters", use_column_width=True)
        else:
            st.info("Cluster visualization plot missing.")
            
    with col_dist_table:
        if os.path.exists(CUSTOMER_SEGMENTS_PATH):
            cust_seg_df = pd.read_csv(CUSTOMER_SEGMENTS_PATH)
            counts = cust_seg_df['Segment'].value_counts()
            pcts = cust_seg_df['Segment'].value_counts(normalize=True) * 100
            dist_summary = pd.DataFrame({
                "Count": counts,
                "Percentage (%)": pcts.round(2)
            })
            st.table(dist_summary)
        else:
            st.info("Segment database not found.")
