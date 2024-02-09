# import streamlit as st
# import requests
# from PIL import Image
# from io import BytesIO
# import numpy as np
# from sklearn.cluster import KMeans
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import Model

# # Function to fetch electronics images from Unsplash
# def fetch_images(query, api_key, per_page=20):
#     url = f"https://api.unsplash.com/search/photos"
#     headers = {"Authorization": f"Client-ID {api_key}"}
#     params = {"query": query, "per_page": per_page}
#     response = requests.get(url, headers=headers, params=params)
#     return [(item['urls']['regular'], item['alt_description']) for item in response.json()['results']]

# # Function to preprocess and download images
# def preprocess_image(img_url):
#     response = requests.get(img_url)
#     img = Image.open(BytesIO(response.content)).convert('RGB')
#     img = img.resize((224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     return preprocess_input(img_array)

# # Extract features from each image
# def extract_features(image_urls):
#     model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
#     features = []
#     for url in image_urls:
#         img_array = preprocess_image(url)
#         if img_array is not None:
#             features.append(model.predict(img_array).flatten())
#     return np.array(features)

# # Cluster the images based on extracted features
# def cluster_images(features, n_clusters=5):
#     if len(features) == 0:
#         return np.array([])
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     return kmeans.fit_predict(features)

# # Display images in Streamlit, organized by cluster
# def display_clusters(image_urls_descriptions, labels):
#     if len(labels) == 0:
#         st.write("No images were clustered.")
#         return

#     unique_labels = set(labels)
#     for label in unique_labels:
#         st.subheader(f'Cluster {label + 1}')
#         cluster_indices = [idx for idx, cluster_label in enumerate(labels) if cluster_label == label]
        
#         # Calculate the number of columns for the grid
#         n_cols = 3  # You can adjust the number of columns based on your preference
#         n_rows = (len(cluster_indices) + n_cols - 1) // n_cols
        
#         for row_idx in range(n_rows):
#             cols = st.columns(n_cols)
#             for col_idx in range(n_cols):
#                 img_idx = row_idx * n_cols + col_idx
#                 if img_idx < len(cluster_indices):
#                     idx = cluster_indices[img_idx]
#                     url, description = image_urls_descriptions[idx]
#                     # Using the column's image method to display the image
#                     cols[col_idx].image(url, caption=description, width=100)  # Adjust width as needed


# def main():
#     st.title("Electronics Image Clustering from Unsplash")
#     api_key = 'MLLiwJrCRyuQwdNDEK5TAyazn8JSqe0IYtFh3T2S4eU'  # Replace with your actual API key
#     query = "electronics"
#     n_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=5)

#     if st.sidebar.button('Fetch and Cluster Images'):
#         with st.spinner('Fetching images...'):
#             image_urls_descriptions = fetch_images(query, api_key)
#             image_urls = [url for url, _ in image_urls_descriptions]
        
#         with st.spinner('Extracting features...'):
#             features = extract_features(image_urls)
        
#         if len(features) > 0:
#             with st.spinner('Clustering images...'):
#                 labels = cluster_images(features, n_clusters=n_clusters)
        
#             st.success('Done!')
#             display_clusters(image_urls_descriptions, labels)
#         else:
#             st.error("No features were extracted. Check if the images were correctly fetched and processed.")

# if __name__ == "__main__":
#     main()



import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Function to fetch images from Unsplash
def fetch_images(query, api_key, per_page=20):
    url = f"https://api.unsplash.com/search/photos"
    headers = {"Authorization": f"Client-ID {api_key}"}
    params = {"query": query, "per_page": per_page}
    response = requests.get(url, headers=headers, params=params)
    return [(item['urls']['regular'], item['alt_description']) for item in response.json()['results']]

# Function to preprocess and download images
def preprocess_image(img_url):
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Extract features from each image
def extract_features(image_urls):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    features = []
    for url in image_urls:
        img_array = preprocess_image(url)
        features.append(model.predict(img_array).flatten())
    return np.array(features)

# Cluster the images based on extracted features
def cluster_images(features, n_clusters=5):
    if len(features) == 0:
        return np.array([])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(features)

# Display images in Streamlit, organized by cluster
def display_clusters(image_urls_descriptions, labels):
    if len(labels) == 0:
        st.write("No images were clustered.")
        return

    unique_labels = set(labels)
    for label in unique_labels:
        st.subheader(f'Cluster {label + 1}')
        cluster_indices = [idx for idx, cluster_label in enumerate(labels) if cluster_label == label]
        n_cols = 3
        n_rows = (len(cluster_indices) + n_cols - 1) // n_cols

        for row_idx in range(n_rows):
            cols = st.columns(n_cols)
            for col_idx in range(n_cols):
                img_idx = row_idx * n_cols + col_idx
                if img_idx < len(cluster_indices):
                    idx = cluster_indices[img_idx]
                    url, description = image_urls_descriptions[idx]
                    cols[col_idx].image(url, caption=description, width=100)

def main():
    st.title("Fashion Image Clustering from Unsplash")
    api_key = 'MLLiwJrCRyuQwdNDEK5TAyazn8JSqe0IYtFh3T2S4eU'  # Replace with your Unsplash API key
    query = "Beauty and Personal Care"
    n_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=5)

    if st.sidebar.button('Fetch and Cluster Images'):
        with st.spinner('Fetching images...'):
            image_urls_descriptions = fetch_images(query, api_key)
            image_urls = [url for url, _ in image_urls_descriptions]
        
        with st.spinner('Extracting features...'):
            features = extract_features(image_urls)
        
        if len(features) > 0:
            with st.spinner('Clustering images...'):
                labels = cluster_images(features, n_clusters=n_clusters)
        
            st.success('Done!')
            display_clusters(image_urls_descriptions, labels)
        else:
            st.error("No features were extracted. Check if the images were correctly fetched and processed.")

if __name__ == "__main__":
    main()
