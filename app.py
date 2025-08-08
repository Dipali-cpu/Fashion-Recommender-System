import streamlit as st
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from PIL import Image

# Load features and filenames
with open("resnet_features.pkl", "rb") as f:
    feature_list = pickle.load(f)

with open("resnet_filenames.pkl", "rb") as f:
    filenames = pickle.load(f)

# Fit Nearest Neighbors
feature_list = np.array(feature_list)
neighbors = NearestNeighbors(n_neighbors=6, algorithm="brute", metric="euclidean")
neighbors.fit(feature_list)

# Streamlit UI
st.title("Fashion Recommender System")
st.write("Upload an image and get similar fashion recommendations!")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Extract features for uploaded image
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

    model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    query_features = model.predict(img_array, verbose=0).flatten()

    # Find similar images
    distances, indices = neighbors.kneighbors([query_features])

    st.subheader("Similar Recommendations:")
    cols = st.columns(5)
    for col, idx in zip(cols, indices[0][1:]):  # skip the uploaded image itself
        col.image(filenames[idx], use_container_width=True)
