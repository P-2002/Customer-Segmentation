import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load scaler, PCA, and KMeans model
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

# Display cluster centroids
centers_scaled = kmeans.cluster_centers_
centers_orig = scaler.inverse_transform(centers_scaled)
cluster_profile = pd.DataFrame(centers_orig, columns=['Income', 'Total_Spending'])
cluster_profile.index = [f"Segment {i}" for i in range(len(centers_orig))]

st.title("Customer Segmentation — KMeans Predictor")

st.write("### Cluster Centroids (Original Scale)")
st.dataframe(cluster_profile.round(2))

# Inputs
income = st.number_input("Income", min_value=0.0, step=1000.0, value=50000.0, format="%.2f")
spending = st.number_input("Total_Spending", min_value=0.0, step=100.0, value=1000.0, format="%.2f")

if st.button("Predict segment"):
    features = np.array([[income, spending]])
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)
    segment = kmeans.predict(features_pca)[0]
    
    # Optional label remapping:
    segment_map = {0: 1, 1: 0}  # Swap 0 and 1 for display
    display_segment = segment_map.get(segment, segment)
    
    st.success(f"Predicted Segment: {display_segment}")
