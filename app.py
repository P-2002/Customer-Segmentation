
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("Customer Segmentation")
st.markdown("Upload a CSV of your customer data, select numeric features, then run PCA + KMeans to visualize clusters.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("Preview of dataset (first 5 rows):")
    st.dataframe(df.head())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found in the uploaded CSV. Please upload a dataset with numeric features.")
    else:
        features = st.multiselect("Select numeric features to use", numeric_cols, default=numeric_cols[:5])
        n_components = st.slider("PCA components", min_value=2, max_value=min(5, len(features)), value=2)
        n_clusters = st.slider("Number of KMeans clusters", min_value=2, max_value=10, value=3)
        run = st.button("Run PCA + KMeans")

        if run and features:
            X = df[features].dropna()
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            pca = PCA(n_components=n_components, random_state=42)
            Xp = pca.fit_transform(Xs)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(Xp)
            sil = silhouette_score(Xp, labels)

            st.write(f"Silhouette score: **{sil:.3f}**")
            res_df = pd.DataFrame(Xp, columns=[f"PC{i+1}" for i in range(Xp.shape[1])])
            res_df['cluster'] = labels
            res_df['index'] = X.index

            st.write("PCA reduced data (first 5 rows):")
            st.dataframe(res_df.head())

            fig, ax = plt.subplots(figsize=(6,5))
            ax.scatter(res_df['PC1'], res_df['PC2'], s=40)
            for i, txt in enumerate(res_df['cluster']):
                ax.annotate(txt, (res_df['PC1'].iat[i], res_df['PC2'].iat[i]), fontsize=8)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title('PCA (PC1 vs PC2) with cluster labels')
            st.pyplot(fig)

            st.markdown("### Cluster counts")
            st.write(res_df['cluster'].value_counts())

            st.download_button("Download clustered results (CSV)", data=res_df.to_csv(index=False), file_name="clustered_results.csv")

else:
    st.info("Upload a CSV to begin.")

