import os
import json
import glob
import google.generativeai as genai
from PIL import Image
import shutil
from datetime import datetime
import tensorflow as tf
import numpy as np

# Configuration
QUEUE_DIR = "analysis/queue"
PROCESSED_DIR = "analysis/processed"
TRAINING_DATA_DIR = "analysis/training_data"
CATEGORY_MAPPING = {0: "Organic", 1: "Recyclable"}
CONFIG_FILE = "config.json"

# Create subdirectories for processed images
PROCESSED_ORGANIC_DIR = os.path.join(PROCESSED_DIR, "organic")
PROCESSED_RECYCLABLE_DIR = os.path.join(PROCESSED_DIR, "recyclable")

def load_config():
    """Load configuration from config file"""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        else:
            # Create default config file
            config = {
                "gemini_api_key": "your-api-key-here"
            }
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Created config file: {CONFIG_FILE}")
            print("Please update the API key in the config file and run the script again.")
            return None
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def setup_gemini():
    """Setup Gemini API with your API key"""
    # First try to get API key from environment
    api_key = os.getenv('GEMINI_API_KEY')
    
    # If not in environment, try to get from config file
    if not api_key:
        config = load_config()
        if config and config.get('gemini_api_key'):
            api_key = config['gemini_api_key']
            # Set it in environment for this session
            os.environ['GEMINI_API_KEY'] = api_key
    
    if not api_key or api_key == "your-api-key-here":
        raise ValueError("Please set GEMINI_API_KEY in config.json or as an environment variable")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    return model

def analyze_image_with_gemini(model, image_path):
    """Analyze image using Gemini API"""
    try:
        # Load and prepare image
        image = Image.open(image_path)
        
        # Prepare prompt
        prompt = """Analyze this image and determine if it shows organic waste or recyclable material.
        Consider the following:
        1. Material type (paper, plastic, food, etc.)
        2. Condition (clean, contaminated, etc.)
        3. Recycling guidelines
        
        Respond with:
        1. Category (Organic/Recyclable)
        2. Confidence (0-1)
        3. Detailed explanation
        """
        
        # Get response from Gemini
        response = model.generate_content([prompt, image])
        response.resolve()
        
        # Parse response
        # Note: You might need to adjust parsing based on actual response format
        lines = response.text.split('\n')
        category = None
        confidence = None
        explanation = []
        
        for line in lines:
            if line.startswith('1. Category:'):
                category = line.split(': ')[1].strip()
            elif line.startswith('2. Confidence:'):
                try:
                    confidence = float(line.split(': ')[1].strip())
                except ValueError:
                    confidence = 0.5  # Default confidence if parsing fails
            elif line.startswith('3. Detailed explanation:'):
                continue
            elif line.strip():
                explanation.append(line.strip())
        
        if not category or confidence is None:
            print(f"Warning: Could not parse response properly. Response was:\n{response.text}")
            return None
        
        return {
            'category': category,
            'confidence': confidence,
            'explanation': '\n'.join(explanation)
        }
    except Exception as e:
        print(f"Error analyzing image with Gemini: {e}")
        return None

def process_queue():
    """Process all images in the queue"""
    # Setup Gemini
    try:
        gemini_model = setup_gemini()
    except Exception as e:
        print(f"Failed to setup Gemini: {e}")
        return
    
    # Create necessary directories
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(PROCESSED_ORGANIC_DIR, exist_ok=True)
    os.makedirs(PROCESSED_RECYCLABLE_DIR, exist_ok=True)
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    
    # Get all images in queue
    image_files = glob.glob(os.path.join(QUEUE_DIR, "*.jpg"))
    
    for image_path in image_files:
        try:
            print(f"\nProcessing {image_path}...")
            
            # Get corresponding metadata
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            metadata_path = os.path.join(QUEUE_DIR, f"{base_name}_metadata.json")
            
            if not os.path.exists(metadata_path):
                print(f"No metadata found for {image_path}")
                continue
            
            # Load original metadata
            with open(metadata_path, 'r') as f:
                original_metadata = json.load(f)
            
            # Analyze with Gemini
            analysis_result = analyze_image_with_gemini(gemini_model, image_path)
            
            if analysis_result:
                # Create new metadata with analysis results
                new_metadata = {
                    **original_metadata,
                    'gemini_analysis': analysis_result,
                    'analysis_timestamp': datetime.now().isoformat()
                }
                
                # Determine target directory based on category
                category = analysis_result['category'].lower()
                if category == 'organic':
                    target_dir = PROCESSED_ORGANIC_DIR
                elif category == 'recyclable':
                    target_dir = PROCESSED_RECYCLABLE_DIR
                else:
                    print(f"Unknown category: {category}, using default processed directory")
                    target_dir = PROCESSED_DIR
                
                # Move to appropriate processed subdirectory
                processed_image_path = os.path.join(target_dir, os.path.basename(image_path))
                processed_metadata_path = os.path.join(target_dir, f"{base_name}_metadata.json")
                
                shutil.move(image_path, processed_image_path)
                with open(processed_metadata_path, 'w') as f:
                    json.dump(new_metadata, f, indent=2)
                
                # If confidence is high enough, add to training data
                if analysis_result['confidence'] >= 0.85:
                    category_dir = os.path.join(TRAINING_DATA_DIR, category)
                    os.makedirs(category_dir, exist_ok=True)
                    
                    training_image_path = os.path.join(
                        category_dir,
                        f"{base_name}_verified.jpg"
                    )
                    shutil.copy2(processed_image_path, training_image_path)
                    print(f"Added to training data: {training_image_path}")
                
                print(f"Processed and moved to: {processed_image_path}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    process_queue() 