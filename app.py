import streamlit as st
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
import pickle
from PIL import Image
from sklearn.neighbors import NearestNeighbors

st.title('Fashion Recommendation System')

upload_folder = 'uploads'
os.makedirs(upload_folder, exist_ok=True)

feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('file_names.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join(upload_folder, uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.read())  # Use .read() instead of .getbuffer()
        return 1
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalised_result = result / norm(result)
    return normalised_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])
    return indices[0]

st.text("--Harikrishna")

uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        st.success("File uploaded successfully!")
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        features = feature_extraction(os.path.join(upload_folder, uploaded_file.name), model)
        indices = recommend(features, feature_list)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(filenames[indices[0]])
        with col2:
            st.image(filenames[indices[1]])
        with col3:
            st.image(filenames[indices[2]])
        with col4:
            st.image(filenames[indices[3]])
        with col5:
            st.image(filenames[indices[4]])
    else:
        st.warning("Some error occurred in file upload")
