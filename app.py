import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

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
st.markdown(
    """
    <div style="background-color:#0d6efd;padding:10px;border-radius:10px">
    <h1 style="color:white;text-align:center;">Nail Disease Prediction</h1>
    </div>
    """, unsafe_allow_html=True
)

st.markdown("<h4 style='color:gray;text-align:center;'>Upload an image to predict the nail disease</h4>", unsafe_allow_html=True)

# Upload image
uploaded_image = st.file_uploader("Choose a nail image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict the class
    img_array = load_and_preprocess_image(uploaded_image)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    
    # Display the prediction with a styled box
    st.markdown(
        f"""
        <div style="background-color:#d4edda;padding:20px;border-radius:10px;margin-top:20px;">
        <h2 style="color:green;text-align:center;">Prediction: {predicted_class}</h2>
        </div>
        """, unsafe_allow_html=True
    )
else:
    st.warning("Please upload an image to get a prediction.")

# Footer
st.markdown(
    """
    <div style="background-color:#f8f9fa;padding:10px;border-radius:10px;margin-top:20px;text-align:center;">
    <p>Powered by Machine Learning | Developed by [Your Name]</p>
    </div>
    """, unsafe_allow_html=True
)
