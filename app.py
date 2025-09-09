import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Potato Disease Detection",
    page_icon="🥔",
    layout="wide"
)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    """Loads the trained Keras model from the .h5 file."""
    try:
        model = tf.keras.models.load_model('potato_disease_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- CLASS NAMES ---
# Make sure this order matches the training order from your notebook
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# --- UI DESIGN ---
st.title("🥔 Potato Leaf Disease Detection")
st.markdown("An AI-powered tool to help farmers identify diseases in potato leaves early.")

st.sidebar.title("About")
st.sidebar.info(
    "This application uses a deep learning model to predict whether a potato leaf is "
    "healthy or suffering from Early Blight or Late Blight. "
    "Upload an image of a potato leaf to get a diagnosis."
    "\n\n**Developed by:** Sarbjot Singh"
)

# --- FILE UPLOADER AND PREDICTION LOGIC ---
uploaded_file = st.file_uploader("Choose a potato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    # THIS IS THE CORRECTED LINE:
    st.image(image, caption='Uploaded Image.', use_container_width=True)

    # Prediction button
    if st.button('Diagnose Disease'):
        if model is not None:
            # Pre-process the image for the model
            image = image.resize((224, 224))
            img_array = np.array(image)
            # Check if the image has an alpha channel and remove it
            if img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            with st.spinner('Analyzing the image...'):
                # Make prediction
                predictions = model.predict(img_array)
                predicted_class = class_names[np.argmax(predictions[0])]
                confidence = round(100 * (np.max(predictions[0])), 2)

            # Display the result
            st.success(f"**Diagnosis:** {predicted_class.replace('___', ' ')}")
            st.info(f"**Confidence:** {confidence}%")
        else:
            st.warning("Model is not loaded. Please check the model file.")
else:
    st.info("Please upload an image to get started.")

