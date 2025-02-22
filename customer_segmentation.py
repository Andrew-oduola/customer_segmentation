# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import pickle
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stMarkdown {
            font-size: 18px;
        }
        .created-by {
            font-size: 20px;
            font-weight: bold;
            color: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🛍️ Customer Segmentation App")
st.markdown("This app uses K-Means clustering to segment customers based on their Annual Income and Spending Score.")

# Sidebar
with st.sidebar:
    st.markdown('<p class="created-by">Created by Andrew O.A.</p>', unsafe_allow_html=True)
    
    # Load and display profile picture
    try:
        profile_pic = Image.open("prof.jpeg")  # Replace with your image file path
        st.image(profile_pic, caption="Andrew O.A.", use_container_width=True, output_format="JPEG")
    except:
        st.warning("Profile image not found.")

    st.title("About")
    st.info("This app uses K-Means clustering to group customers into segments based on their income and spending habits.")
    st.markdown("[GitHub](https://github.com/Andrew-oduola) | [LinkedIn](https://linkedin.com/in/andrew-oduola-django-developer)")

# Option to use the default model
use_default_model = st.checkbox("Use Default Pre-Trained Model", value=False)

if use_default_model:
    st.info("Using the default pre-trained model for customer segmentation.")
    
    # Load the default dataset
    default_data = pd.read_csv("Mall_Customers.csv")  # Ensure this file is in the same directory
    st.subheader("Default Dataset Preview")
    st.write(default_data.head())

    # Extract relevant columns
    x = default_data.iloc[:, [3, 4]].values

    # Load the pre-trained model
    try:
        with open('customer_cluster.pkl', 'rb') as file:
            kmeans_default = pickle.load(file)
        st.success("Default model loaded successfully.")

        # Make predictions using the default model
        Y_default = kmeans_default.predict(x)

        # Visualize the clusters
        st.subheader("Customer Segmentation Clusters (Default Model)")
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ['green', 'red', 'yellow', 'blue', 'black', 'purple', 'orange', 'pink', 'brown', 'gray']
        for i in range(kmeans_default.n_clusters):
            ax.scatter(x[Y_default == i, 0], x[Y_default == i, 1], s=50, c=colors[i], label=f'Cluster {i+1}')
        
        # Plot the centroids
        ax.scatter(kmeans_default.cluster_centers_[:, 0], kmeans_default.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')
        ax.set_title('Customer Groups (Default Model)')
        ax.set_xlabel('Annual Income (k$)')
        ax.set_ylabel('Spending Score (1-100)')
        ax.legend()
        st.pyplot(fig)

    except FileNotFoundError:
        st.error("Default model file 'customer_cluster.pkl' not found. Please ensure the file is in the same directory.")
else:
    # File uploader for CSV
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load the data
        customer_data = pd.read_csv(uploaded_file)
        
        # Display the first few rows of the dataset
        st.subheader("Dataset Preview")
        st.write(customer_data.head())

        # Check for required columns
        required_columns = ["Annual Income (k$)", "Spending Score (1-100)"]
        if all(col in customer_data.columns for col in required_columns):
            st.success("Required columns found in the dataset.")
            
            # Extract relevant columns
            x = customer_data.iloc[:, [3, 4]].values

            # Elbow Method to find the optimal number of clusters
            st.subheader("Elbow Method to Determine Optimal Clusters")
            wcss = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
                kmeans.fit(x)
                wcss.append(kmeans.inertia_)

            # Plot the Elbow Graph
            fig, ax = plt.subplots()
            sns.set()
            ax.plot(range(1, 11), wcss, marker='o')
            ax.set_title('The Elbow Point Graph')
            ax.set_xlabel('Number of Clusters')
            ax.set_ylabel('WCSS')
            st.pyplot(fig)

            # Optimal number of clusters
            optimal_clusters = st.slider("Select the number of clusters", min_value=2, max_value=10, value=5)

            # Train the K-Means model
            kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=0)
            Y = kmeans.fit_predict(x)

            # Visualize the clusters
            st.subheader("Customer Segmentation Clusters")
            fig, ax = plt.subplots(figsize=(8, 8))
            colors = ['green', 'red', 'yellow', 'blue', 'black', 'purple', 'orange', 'pink', 'brown', 'gray']
            for i in range(optimal_clusters):
                ax.scatter(x[Y == i, 0], x[Y == i, 1], s=50, c=colors[i], label=f'Cluster {i+1}')
            
            # Plot the centroids
            ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')
            ax.set_title('Customer Groups')
            ax.set_xlabel('Annual Income (k$)')
            ax.set_ylabel('Spending Score (1-100)')
            ax.legend()
            st.pyplot(fig)

            # Save the model
            if st.button("Save Model"):
                with open('customer_cluster.pkl', 'wb') as file:
                    pickle.dump(kmeans, file)
                st.success("Model saved successfully as 'customer_cluster.pkl'.")

        else:
            st.error(f"The dataset must contain the following columns: {required_columns}")
    else:
        st.info("Please upload a CSV file to get started.")