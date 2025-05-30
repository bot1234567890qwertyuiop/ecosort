from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import uuid
from datetime import datetime
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
from predict import predict_image, load_model, IMG_HEIGHT, IMG_WIDTH, CATEGORY_MAPPING

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
PREDICTIONS_FOLDER = 'static/predictions'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)

# Load model at startup
model = load_model("/Users/tracy/Desktop/EcoSort.0.2.4/models/garbage_classifier_best.h5")

def save_prediction_image(image, filename):
    """Save the prediction visualization"""
    output_path = os.path.join(PREDICTIONS_FOLDER, f"{filename}_prediction.png")
    image.save(output_path)
    return f"predictions/{filename}_prediction.png"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Generate unique filename
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        filepath = os.path.join(UPLOAD_FOLDER, f"{filename}.jpg")
        
        # Save uploaded file
        file.save(filepath)

        # Make prediction
        predicted_class, confidence = predict_image(model, filepath, display=True)
        
        # Get prediction image path
        prediction_image = f"{filename}_prediction.png"
        
        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': float(confidence),
            'image_path': f"uploads/{filename}.jpg",
            'prediction_image': f"predictions/{prediction_image}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000) 