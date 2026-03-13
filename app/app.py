import os
import io
import base64
import sys
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

# Add parent directory to path to import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.efficientnet_colorizer import build_efficientnet_colorizer

app = Flask(__name__)

# Global model initialization
MODEL_PATH = os.path.join('..', 'models', 'saved_models', 'efficientnet_best_model.h5')
print("Initializing model...")
model = build_efficientnet_colorizer()

if os.path.exists(MODEL_PATH):
    model(tf.zeros((1, 224, 224, 1)))  # Build dummy input
    model.load_weights(MODEL_PATH)
    print("Model weights loaded successfully.")
else:
    print(f"Warning: Model weights not found at {MODEL_PATH}. Using untrained model.")

def process_image(img_bytes):
    img = tf.io.decode_image(img_bytes, channels=3)
    original_shape = tf.shape(img)[:2]
    
    img_resized = tf.image.resize(img, (224, 224))
    img_resized = img_resized / 255.0
    
    lab_img = tfio.experimental.color.rgb_to_lab(img_resized)
    L = lab_img[..., 0:1] / 100.0
    
    L_input = tf.expand_dims(L, axis=0)
    pred_ab_normalized = model(L_input, training=False)[0]
    
    pred_ab = pred_ab_normalized * 128.0
    L_denorm = lab_img[..., 0:1]
    
    lab_pred = tf.concat([L_denorm, pred_ab], axis=-1)
    rgb_pred = tfio.experimental.color.lab_to_rgb(lab_pred)
    
    rgb_pred = tf.image.resize(rgb_pred, original_shape)
    rgb_pred = tf.clip_by_value(rgb_pred, 0.0, 1.0)
    rgb_pred = tf.cast(rgb_pred * 255.0, tf.uint8)
    
    # Convert prediction to base64
    out_img = Image.fromarray(rgb_pred.numpy())
    buffered = io.BytesIO()
    out_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return img_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/colorize', methods=['POST'])
def colorize():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    try:
        img_bytes = file.read()
        
        # Original Image Base64
        original_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        buffered_orig = io.BytesIO()
        original_img.save(buffered_orig, format="JPEG")
        orig_base64 = base64.b64encode(buffered_orig.getvalue()).decode("utf-8")
        
        # Colorized Image Base64
        color_base64 = process_image(img_bytes)
        
        return jsonify({
            'original': f"data:image/jpeg;base64,{orig_base64}",
            'colorized': f"data:image/jpeg;base64,{color_base64}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
