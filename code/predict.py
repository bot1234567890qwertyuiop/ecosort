import os
import glob
from pathlib import Path
from pillow_heif import register_heif_opener
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import json
from datetime import datetime
import shutil

# Define constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
CATEGORY_MAPPING = {0: "Organic", 1: "Recyclable"}
MODEL_PATH = "/Users/tracy/Desktop/EcoSort.0.2.4/models/garbage_classifier_best.h5"  # Using best model
QUEUE_DIR = "/Users/tracy/Desktop/EcoSort.0.2.4/analysis/queue"
PREDICTIONS_DIR = "/Users/tracy/Desktop/EcoSort.0.2.4/predictions"
INPUT_DIR = "/Users/tracy/Desktop/EcoSort.0.2.4/predict_queue"
TEMP_DIR = "/Users/tracy/Desktop/EcoSort.0.2.4/temp_jpg"
PREDICTED_DIR = "/Users/tracy/Desktop/EcoSort.0.2.4/predicted"
QUEUE_FILE = os.path.join(QUEUE_DIR, "low_confidence_queue.json")


# Initialize queue
prediction_queue = deque()

def save_queue():
    """Save the queue to a JSON file"""
    os.makedirs(QUEUE_DIR, exist_ok=True)
    queue_data = list(prediction_queue)
    with open(QUEUE_FILE, 'w') as f:
        json.dump(queue_data, f, indent=2)

def load_queue():
    """Load the queue from JSON file if it exists"""
    global prediction_queue
    if os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE, 'r') as f:
            prediction_queue = deque(json.load(f))
        print(f"Loaded {len(prediction_queue)} items from queue")

def convert_to_jpg(input_path, output_path=None):
    """Convert any image format to JPG"""
    try:
        # Register HEIF opener for HEIC files
        register_heif_opener()
        
        # Open and convert image
        image = Image.open(input_path)
        
        # Convert to RGB if necessary (for PNG with transparency)
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
            image = background
        
        # Save as JPG
        if output_path is None:
            output_path = os.path.splitext(input_path)[0] + ".jpg"
        image.save(output_path, "JPEG", quality=95)
        print(f"Converted {input_path} to JPG")
        return output_path
    except Exception as e:
        print(f"Error converting {input_path}: {str(e)}")
        return None

def convert_all_images(input_dir, output_dir):
    """Convert all images in input directory to JPG"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Supported image formats
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.HEIC', '*.heic')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    converted_files = []
    for image_file in image_files:
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        output_path = os.path.join(output_dir, f"{base_name}.jpg")
        if convert_to_jpg(image_file, output_path):
            converted_files.append(output_path)
    
    print(f"Converted {len(converted_files)} images to JPG")
    return converted_files

def load_model(model_path):
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_image(model, image_path, display=False):
    """Predict the class of a single image"""
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    
    # Get probabilities for each class
    organic_prob = float(predictions[0][0])  # Class 0: Organic
    recyclable_prob = float(predictions[0][1])  # Class 1: Recyclable
    all_probs = [organic_prob, recyclable_prob]
    
    # Get predicted class
    predicted_class = np.argmax(predictions[0])
    predicted_class_name = CATEGORY_MAPPING[predicted_class]
    confidence = float(predictions[0][predicted_class])
    
    # Print results
    print("\nPrediction Results:")
    print(f"Image: {image_path}")

    waste_overwrite_limit = 0.75
    waste = False
    if confidence <= waste_overwrite_limit:
        print("Prediction overwrite --> Waste")
        waste = True
        # Add to queue for review
        queue_item = {
            'image_path': image_path,
            'predicted_class': predicted_class_name,
            'confidence': confidence,
            'probabilities': {
                'organic': organic_prob,
                'recyclable': recyclable_prob
            },
            'timestamp': datetime.now().isoformat()
        }
        prediction_queue.append(queue_item)
        save_queue()
        print(f"Added to review queue. Queue size: {len(prediction_queue)}")
    
    print(f"Predicted class: {predicted_class_name}")
    print(f"Confidence: {confidence:.2f}\n")
    print("Confidence by class:")
    print(f"Organic: {organic_prob:.2f}")
    print(f"Recyclable: {recyclable_prob:.2f}")
    print(f"Waste (overwrite) limit: {waste_overwrite_limit}")
    
    # Display visualization if requested
    if display:
        display_prediction(img, predicted_class_name, waste, confidence, all_probs, image_path)
    
    return predicted_class_name, confidence

def display_prediction(image, predicted_class, waste, confidence, all_probs, image_path):
    """Display the image with prediction results"""
    plt.switch_backend('Agg')
    plt.figure(figsize=(12, 5))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    if waste:
        plt.title(f"Predicted: Waste | original: {predicted_class} \nConfidence: {confidence:.2f}")
    else:
        plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}")
    plt.axis('off')
    
    # Display confidence bars
    plt.subplot(1, 2, 2)
    classes = list(CATEGORY_MAPPING.values())
    plt.bar(classes, all_probs, color=['green', 'blue'])
    plt.title('Confidence by Class')
    plt.ylabel('Confidence')
    plt.ylim(0, 1)
    
    # Save the plot
    plt.tight_layout()
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(PREDICTIONS_DIR, f"{base_name}_prediction.png")
    plt.savefig(output_path)
    plt.close()
    
    print(f"\nPrediction visualization saved to: {output_path}")

def process_all():
    """Process all images in the predict_queue directory"""
    # Create necessary directories
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(QUEUE_DIR, exist_ok=True)
    os.makedirs(PREDICTED_DIR, exist_ok=True)
    
    # Load existing queue
    load_queue()
    
    # Step 1: Convert all images to JPG
    print("\nConverting images to JPG...")
    converted_files = convert_all_images(INPUT_DIR, TEMP_DIR)
    
    if not converted_files:
        print("No images found to process")
        return
    
    # Step 2: Load model
    model = load_model(MODEL_PATH)
    if model is None:
        return
    
    print(f"\nProcessing {len(converted_files)} images...")
    
    # Step 3: Process each image
    for jpg_file in converted_files:
        try:
            print(f"\nProcessing {os.path.basename(jpg_file)}...")
            predict_image(model, jpg_file, display=True)
            
            # Move original file to predicted directory
            original_file = os.path.join(INPUT_DIR, os.path.basename(jpg_file).replace('.jpg', '.HEIC'))
            if os.path.exists(original_file):
                shutil.move(original_file, os.path.join(PREDICTED_DIR, os.path.basename(original_file)))
                print(f"Moved {os.path.basename(original_file)} to predicted directory")
        except Exception as e:
            print(f"Error processing {jpg_file}: {str(e)}")
    
    # Step 4: Clean up temporary files
    print("\nCleaning up temporary files...")
    shutil.rmtree(TEMP_DIR)
    
    # Print queue summary
    if prediction_queue:
        print("\nQueue Summary:")
        print(f"Total items in queue: {len(prediction_queue)}")
        print(f"Queue saved to: {QUEUE_FILE}")

if __name__ == "__main__":
    process_all() 