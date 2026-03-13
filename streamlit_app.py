import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from skimage import color

from models.efficientnet_colorizer import build_efficientnet_colorizer

st.set_page_config(page_title="AI Image Colorizer", layout="wide", page_icon="🎨")

st.title("🎨 AI Image Colorizer")
st.write("Upload a grayscale or black-and-white image and compare Base Model vs Fine-Tuned Model")

IMG_SIZE = 256

# ---------- LOAD MODELS ----------

@st.cache_resource
def load_base_model():
    model = build_efficientnet_colorizer((None, None, 1))
    path = "models/saved_models/efficientnet_best_model.h5"
    if os.path.exists(path):
        try:
            model.load_weights(path, by_name=True, skip_mismatch=True)
        except Exception as e:
            st.warning(f"Error loading Base Model weights: {e}")
    else:
        st.warning(f"Base Model weights not found at {path}.")
    return model

@st.cache_resource
def load_finetuned_model():
    model = build_efficientnet_colorizer((None, None, 1))
    path = "models/saved_models/efficientnet_finetuned.h5"
    if os.path.exists(path):
        try:
            model.load_weights(path, by_name=True, skip_mismatch=True)
        except Exception as e:
            st.warning(f"Error loading Fine-Tuned Model weights: {e}")
    else:
        st.warning(f"Fine-Tuned Model not found at {path}. Run fine-tuning first.")
    return model

base_model = load_base_model()
finetuned_model = load_finetuned_model()

# ---------- COLORIZATION FUNCTION ----------

def colorize(image, model):
    # Resize to standard size for prediction
    img = np.array(image.resize((IMG_SIZE, IMG_SIZE))) / 255.0

    # skimage requires images in [0, 1] range to output properly
    lab = color.rgb2lab(img)

    L = lab[:, :, 0] / 100.0  # Normalize to [0, 1]
    L_input = L.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    pred_ab = model.predict(L_input, verbose=0)[0]
    pred_ab = pred_ab * 128.0

    lab_output = np.zeros((IMG_SIZE, IMG_SIZE, 3))
    lab_output[:, :, 0] = lab[:, :, 0]
    lab_output[:, :, 1:] = pred_ab

    # Convert back to RGB
    rgb = color.lab2rgb(lab_output)

    # Clip values and format as Image
    rgb = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    rgb = rgb.resize(image.size)  # Restore original resolution

    return rgb

# ---------- FILE UPLOAD ----------

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    if st.button("Colorize Image ✨"):
        with st.spinner("Generating colors using standard and finetuned model..."):
            base_result = colorize(image, base_model)
            fine_result = colorize(image, finetuned_model)

        with col2:
            st.subheader("Base Model Output")
            st.image(base_result, use_container_width=True)

        with col3:
            st.subheader("Fine-Tuned Model Output")
            st.image(fine_result, use_container_width=True)

st.markdown("---")
st.markdown("Developed with ❤️ using TensorFlow + EfficientNet Autoencoder")