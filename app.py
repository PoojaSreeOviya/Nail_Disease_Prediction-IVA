import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

# Load the trained model
model = tf.keras.models.load_model('nail_disease_model.keras')

# Define class names
class_names = ['healthy', 'onychomycosis', 'psoriasis']

# Preprocessing function for the uploaded image
def load_and_preprocess_image(image):
    size = (224, 224)  # Image size expected by the model
    image = Image.open(image).convert('RGB')  # Ensure 3 channels
    image = np.array(image)  # Convert to numpy array
    image_resized = cv2.resize(image, size)
    image_array = np.expand_dims(image_resized, axis=0)  # Expand dims for batch processing
    return image_array

# Streamlit app UI
st.title("Nail Disease Prediction")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict the class
    img_array = load_and_preprocess_image(uploaded_image)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    
    # Display the prediction
    st.write(f"Predicted Class: {predicted_class}")
