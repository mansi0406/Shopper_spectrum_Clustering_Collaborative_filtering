
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_option_menu import option_menu

# --- Page Configuration ---
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
)

# ---- (OPTIONAL) Remove this if not styling ----
# st.markdown("", unsafe_allow_html=True)

# --- Load Models and Data ---
@st.cache_data
def load_data():
    with open('kmeans_model.pkl', 'rb') as file:
        kmeans_model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('product_list.pkl', 'rb') as file:
        product_list = pickle.load(file)
    product_similarity_df = pd.read_pickle('product_similarity_df.pkl')
    rfm_df = pd.read_pickle('rfm_df.pkl')
    return kmeans_model, scaler, product_list, product_similarity_df, rfm_df

try:
    kmeans, scaler, product_list, similarity_df, rfm_df = load_data()
except FileNotFoundError:
    st.error("Error: Model or data files not found. Please ensure all .pkl files (including rfm_df.pkl) are in the same directory.")
    st.stop()


# --- Sidebar Navigation Menu ---
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Home", "Clustering", "Recommendation"],
        icons=["house-door-fill", "person-bounding-box", "basket-fill"],
        menu_icon="cast", default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#F0F2F6"},
            "icon": {"color": "#455A64", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#E3F2FD"},
            "nav-link-selected": {"background-color": "#E53935", "color": "white"},
        }
    )

# --- HOME PAGE ---
if selected == "Home":
    st.title("Welcome to Shopper Spectrum 🛒")
    st.write("This application provides tools for customer segmentation and product recommendations based on e-commerce transaction data.")
    st.markdown("- **Clustering:** Predict a customer's segment based on their RFM values.")
    st.markdown("- **Recommendation:** Get the top 5 similar product recommendations.")

# --- CLUSTERING PAGE ---
elif selected == "Clustering":
    st.title("Customer Segmentation")
    st.write("Enter Recency, Frequency, and Monetary values below to predict the customer segment:")

    recency = st.number_input("Recency (days since last purchase)", min_value=0, step=1, value=325)
    frequency = st.number_input("Frequency (number of purchases)", min_value=1, step=1, value=1)
    monetary = st.number_input("Monetary (total spend)", min_value=0.01, format="%.2f", value=100.00)

    if st.button("Predict Segment"):
        input_data = pd.DataFrame({'Recency': [recency], 'Frequency': [frequency], 'Monetary': [monetary]})
        input_log = np.log1p(input_data)
        input_scaled = scaler.transform(input_log)
        predicted_cluster = kmeans.predict(input_scaled)[0]

        rfm_df['Cluster'] = kmeans.labels_
        cluster_summary = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        cluster_summary['Recency_Rank'] = cluster_summary['Recency'].rank(ascending=True)
        cluster_summary['Frequency_Rank'] = cluster_summary['Frequency'].rank(ascending=False)
        cluster_summary['Monetary_Rank'] = cluster_summary['Monetary'].rank(ascending=False)
        cluster_summary['Overall_Score'] = (
            cluster_summary['Recency_Rank'] +
            cluster_summary['Frequency_Rank'] +
            cluster_summary['Monetary_Rank']
        )
        sorted_clusters = cluster_summary.sort_values('Overall_Score', ascending=True)
        label_mapping = {
            sorted_clusters.index[0]: 'High-Value',
            sorted_clusters.index[1]: 'Regular',
            sorted_clusters.index[2]: 'Occasional',
            sorted_clusters.index[3]: 'At-Risk'
        }
        predicted_segment = label_mapping.get(predicted_cluster, "Unknown")

        st.success(f"Predicted cluster number: {predicted_cluster}")
        st.info(f"This customer belongs to: **{predicted_segment} Shopper**")

        st.write("Cluster summary (mean RFM values):")
        st.dataframe(cluster_summary.style.format("{:.2f}"))
        st.write("Label mapping (cluster_number ⇒ segment):")
        st.write(label_mapping)

# --- RECOMMENDATION PAGE ---
elif selected == "Recommendation":
    st.title("Product Recommender")
    st.write("Enter a product name to receive recommendations:")

    selected_product = st.text_input("Enter Product Name", value="GREEN VINTAGE SPOT BEAKER")

    if st.button("Recommend"):
        if selected_product:
            try:
                similar_scores = similarity_df[selected_product].sort_values(ascending=False)
                top_5_recommendations = similar_scores.drop(selected_product).head(5)
                st.write("**Recommended Products:**")
                for product in top_5_recommendations.index:
                    st.write(product)
            except KeyError:
                st.error(f"Product not found. Please try another product name.")
        else:
            st.error("Please enter a product name.")
