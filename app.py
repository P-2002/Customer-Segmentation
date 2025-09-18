import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# -------------------------------
# Streamlit App
# -------------------------------

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation App")

# File upload
uploaded_file = st.file_uploader("Upload your Excel file", type="xlsx")

if uploaded_file:
    # Load Excel file
    df = pd.read_excel(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Select numeric columns for clustering
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.subheader("Select columns for clustering")
    selected_cols = st.multiselect("Choose numeric columns", numeric_cols, default=numeric_cols)

    if selected_cols:
        X = df[selected_cols]

        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA for dimensionality reduction
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # KMeans clustering
        n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_pca)

        # Add cluster column to dataframe
        df['Cluster'] = clusters
        st.subheader("Clustered Data")
        st.dataframe(df.head())

        # Visualization
        st.subheader("Cluster Visualization")
        fig, ax = plt.subplots()
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="Set2", s=100, ax=ax)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title("Customer Segments")
        st.pyplot(fig)
else:
    st.info("Please upload an Excel file to get started.")


