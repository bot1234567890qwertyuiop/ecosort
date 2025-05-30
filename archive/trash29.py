import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
import cv2
import glob
import gc  # For garbage collection
import json
from collections import Counter
import multiprocessing
from tensorflow.keras.applications import ResNet50V2, EfficientNetV2B0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.losses import CategoricalCrossentropy
import datetime
import math
from tqdm import tqdm  # Add tqdm for progress bars

# Define script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(SCRIPT_DIR)

# Get total CPU cores and GPU memory
total_cpu_cores = multiprocessing.cpu_count()
cpu_limit = max(1, total_cpu_cores // 2)  # Use 50% of CPU cores

# Configure GPU memory growth and limit CPU usage
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # Enable memory growth
            tf.config.experimental.set_memory_growth(gpu, True)
            
            # Get GPU memory (typical GPU has 8GB = 8192MB)
            gpu_mem = 8192  # Default assumption
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                if 'memory_limit' in gpu_details:
                    gpu_mem = gpu_details['memory_limit'] // (1024 * 1024)  # Convert to MB
            except:
                pass
            
            # Set memory limit to 50% of total memory
            memory_limit = gpu_mem // 2
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
            )
        print(f"GPU configuration: Limited to {memory_limit}MB per GPU")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Limit CPU threads to 50% of available cores
tf.config.threading.set_inter_op_parallelism_threads(cpu_limit)  # Number of thread pools
tf.config.threading.set_intra_op_parallelism_threads(cpu_limit)  # Threads per pool
print(f"CPU configuration: Limited to {cpu_limit} cores out of {total_cpu_cores} available cores")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32  # Increased from 16 for better gradient estimates
EPOCHS = 150  # Increased from 100 to allow more training time
LEARNING_RATE = 0.001  # Base learning rate for One Cycle Policy
MAX_LR = 0.01  # Maximum learning rate for One Cycle Policy
# Use a hardcoded absolute path to the DATASET directory
DATA_DIR = '/Users/tracy/Desktop/EcoSort/DATASET'
MODEL_SAVE_PATH = 'trash_classifier_model.h5'
CATEGORY_MAPPING_PATH = 'category_mapping.json'
USE_TRANSFER_LEARNING = True
USE_CROSS_VALIDATION = True
USE_CLASS_BALANCING = True
USE_FEATURE_ENGINEERING = True  # Enable feature engineering
USE_MIXUP = True  # Enable Mixup data augmentation
K_FOLDS = 5
USE_EFFICIENTNET = True
CONFIDENCE_THRESHOLD = 0.7
VERBOSE = 1  # Show progress bar
MIXUP_ALPHA = 0.2  # Alpha parameter for Mixup

class OneCycleLR(keras.callbacks.Callback):
    """One Cycle Policy learning rate scheduler."""
    def __init__(self, max_lr, epochs, steps_per_epoch, pct_start=0.3, div_factor=25.0, final_div_factor=1e4):
        super(OneCycleLR, self).__init__()
        self.max_lr = max_lr
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.total_steps = epochs * steps_per_epoch
        self.step_size_up = int(pct_start * self.total_steps)
        self.step_size_down = self.total_steps - self.step_size_up
        self.lr_max = max_lr
        self.lr_start = max_lr / div_factor
        self.lr_min = self.lr_start / final_div_factor
        self.iteration = 0
        
    def on_train_begin(self, logs=None):
        self.iteration = 0
        keras.backend.set_value(self.model.optimizer.lr, self.lr_start)
        
    def on_batch_end(self, batch, logs=None):
        self.iteration += 1
        if self.iteration <= self.step_size_up:
            # Learning rate increase phase
            lr = self.lr_start + (self.lr_max - self.lr_start) * (self.iteration / self.step_size_up)
        else:
            # Learning rate decrease phase
            lr = self.lr_max - (self.lr_max - self.lr_min) * ((self.iteration - self.step_size_up) / self.step_size_down)
        
        keras.backend.set_value(self.model.optimizer.lr, lr)
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = keras.backend.get_value(self.model.optimizer.lr)

def get_category_mapping():
    """Create a mapping between category folders and class indices."""
    # Binary classification: R (Recyclable) = 0, O (Organic) = 1
    return {'R': 0, 'O': 1}

def load_and_preprocess_data():
    """Load and preprocess the dataset with memory-efficient processing."""
    print("\n=== Starting Data Loading and Preprocessing ===")
    
    # Get category mapping
    category_mapping = get_category_mapping()
    num_classes = len(category_mapping)
    print(f"\nTotal number of classes: {num_classes}")
    
    # Initialize lists to store image paths and labels
    image_paths = []
    labels = []
    class_counts = {'R': 0, 'O': 0}
    
    # Process training data with progress tracking
    train_dir = os.path.join(DATA_DIR, 'TRAIN')
    print(f"\nLooking for training data in: {train_dir}")
    print(f"Absolute path: {os.path.abspath(train_dir)}")
    
    if not os.path.exists(train_dir):
        print(f"Directory not found at: {train_dir}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {SCRIPT_DIR}")
        print(f"Workspace directory: {WORKSPACE_DIR}")
        print(f"Data directory: {DATA_DIR}")
        print(f"Directory exists: {os.path.exists(DATA_DIR)}")
        if os.path.exists(DATA_DIR):
            print(f"Contents of DATA_DIR: {os.listdir(DATA_DIR)}")
        raise FileNotFoundError(f"Training directory not found at {train_dir}")
    
    total_processed = 0
    
    print("\nScanning dataset directories...")
    for category in ['R', 'O']:
        category_path = os.path.join(train_dir, category)
        if not os.path.exists(category_path):
            print(f"Warning: Category path {category_path} does not exist")
            continue
            
        # Get all image files in the category
        image_files = glob.glob(os.path.join(category_path, '*.jpg'))
        class_counts[category] = len(image_files)
        
        print(f"\nProcessing {category} category: {len(image_files)} images")
        # Add progress bar for each category
        for img_path in tqdm(image_files, desc=f"Loading {category} images", unit="img", ncols=100):
            image_paths.append(img_path)
            labels.append(category_mapping[category])
            total_processed += 1
            
            # Clear memory periodically
            if total_processed % 5000 == 0:
                gc.collect()
    
    if total_processed == 0:
        raise ValueError("No images found in the training directory")
    
    print("\n=== Dataset Loading Summary ===")
    print("\nImages per category:")
    for category, count in class_counts.items():
        print(f"{category}: {count} images")
    
    # Convert labels to numpy array
    print("\nConverting labels to one-hot encoding...")
    y = np.array(labels)
    y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
    
    # Split into training and validation sets
    print("\nSplitting dataset into training and validation sets...")
    X_paths_train, X_paths_val, y_train, y_val = train_test_split(
        image_paths, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDataset split: {len(X_paths_train)} training images, {len(X_paths_val)} validation images")
    
    # Calculate class weights for balanced training
    class_weight_dict = None
    if USE_CLASS_BALANCING:
        print("\nCalculating class weights for balanced training...")
        y_indices = np.argmax(y_train, axis=1)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_indices),
            y=y_indices
        )
        class_weight_dict = dict(enumerate(class_weights))
        print(f"Class weights: {class_weight_dict}")
    
    print("\n=== Data Loading Complete ===")
    return X_paths_train, X_paths_val, y_train, y_val, num_classes, category_mapping, class_weight_dict

def apply_feature_engineering(img):
    """Apply enhanced feature engineering to the image."""
    # Ensure image is in the correct format (float32, 0-1 range)
    img_float = img.astype('float32')
    
    # Convert to grayscale for edge detection (0-1 range)
    gray = cv2.cvtColor(img_float, cv2.COLOR_RGB2GRAY)
    
    # Convert to 8-bit format for Canny edge detection (0-255 range)
    gray_8bit = (gray * 255).astype('uint8')
    
    # Apply edge detection with multiple thresholds
    edges_low = cv2.Canny(gray_8bit, 50, 150)
    edges_high = cv2.Canny(gray_8bit, 100, 200)
    
    # Convert back to float32 (0-1 range)
    edges_low = edges_low.astype('float32') / 255.0
    edges_high = edges_high.astype('float32') / 255.0
    
    # Apply histogram equalization for better contrast
    # Convert to 8-bit for equalization
    gray_8bit_eq = cv2.equalizeHist(gray_8bit)
    # Convert back to float32 (0-1 range)
    equalized = gray_8bit_eq.astype('float32') / 255.0
    
    # Apply adaptive histogram equalization for better local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    adaptive_eq_8bit = clahe.apply(gray_8bit)
    # Convert back to float32 (0-1 range)
    adaptive_eq = adaptive_eq_8bit.astype('float32') / 255.0
    
    # Apply Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply additional preprocessing
    # Sharpen the image
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(img_float, -1, kernel)
    
    # Create enhanced RGB image by incorporating multiple features
    enhanced_img = img_float.copy()
    
    # Red channel: Original + edges_low + sharpened
    enhanced_img[:,:,0] = (enhanced_img[:,:,0] + edges_low + sharpened[:,:,0]) / 3
    
    # Green channel: Original + edges_high + sharpened
    enhanced_img[:,:,1] = (enhanced_img[:,:,1] + edges_high + sharpened[:,:,1]) / 3
    
    # Blue channel: Original + adaptive equalization + sharpened
    enhanced_img[:,:,2] = (enhanced_img[:,:,2] + adaptive_eq + sharpened[:,:,2]) / 3
    
    return enhanced_img

def create_data_generator(image_paths, labels, batch_size, is_training=True, class_weights=None):
    """Create a memory-efficient data generator that loads images on-the-fly."""
    print(f"\nCreating data generator with {len(image_paths)} images")
    num_samples = len(image_paths)
    
    # Create data augmentation for training
    if is_training:
        print("Creating training data generator with augmentation")
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
    else:
        print("Creating validation data generator without augmentation")
        datagen = keras.preprocessing.image.ImageDataGenerator()
    
    def generator():
        while True:
            # Generate batch indices
            indices = np.random.permutation(num_samples)
            
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_paths = [image_paths[j] for j in batch_indices]
                batch_labels = labels[batch_indices]
                
                # Load and preprocess batch of images
                batch_images = []
                batch_weights = []
                
                for img_path in batch_paths:
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            print(f"Warning: Could not load image {img_path}")
                            continue
                            
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
                        img = img.astype('float32') / 255.0
                        
                        if USE_FEATURE_ENGINEERING:
                            img = apply_feature_engineering(img)
                            
                        batch_images.append(img)
                        
                        # Apply class weights if provided
                        if class_weights is not None:
                            label_idx = np.argmax(batch_labels[len(batch_images)-1])
                            batch_weights.append(class_weights[label_idx])
                        else:
                            batch_weights.append(1.0)
                            
                    except Exception as e:
                        print(f"Error processing image {img_path}: {str(e)}")
                        continue
                
                if not batch_images:
                    continue
                    
                batch_images = np.array(batch_images)
                batch_weights = np.array(batch_weights)
                
                # Apply data augmentation if training
                if is_training:
                    batch_images = datagen.standardize(batch_images)
                
                yield batch_images, batch_labels, batch_weights
    
    print("Data generator created successfully")
    return generator(), num_samples // batch_size

def create_transfer_model(num_classes):
    """Create a model using transfer learning with a pre-trained network."""
    if USE_EFFICIENTNET:
        # Use EfficientNetV2B0 as the base model (often better for smaller datasets)
        base_model = EfficientNetV2B0(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
        )
    else:
        # Use ResNet50V2 as the base model
        base_model = ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
        )
    
    # Freeze the base model layers except the last few blocks
    base_model.trainable = True
    for layer in base_model.layers[:-40]:  # Unfreeze more layers (40 instead of 30)
        layer.trainable = False
    
    # Create the model
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Apply the base model
    x = base_model(inputs, training=True)  # Enable training for fine-tuning
    
    # Add classification layers with stronger regularization
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)  # Increased dropout
    
    # Add an additional dense layer with more units
    x = Dense(2048, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Original dense layers
    x = Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Add a final dense layer with fewer units before the output
    x = Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model, base_model

def create_simple_model(num_classes):
    """Create a simpler CNN model for waste classification."""
    # Add L2 regularization with higher strength
    regularizer = keras.regularizers.l2(0.05)  # Increased regularization
    
    # Create a simpler model
    model = keras.Sequential([
        # First Convolutional Block
        keras.layers.Conv2D(32, (3, 3), activation='relu', 
                           kernel_regularizer=regularizer,
                           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.5),  # Increased dropout
        
        # Second Convolutional Block
        keras.layers.Conv2D(64, (3, 3), activation='relu',
                           kernel_regularizer=regularizer),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.5),
        
        # Third Convolutional Block
        keras.layers.Conv2D(64, (3, 3), activation='relu',
                           kernel_regularizer=regularizer),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Dropout(0.5),
        
        # Flatten and Dense Layers
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu',  # Reduced from 64
                          kernel_regularizer=regularizer),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.6),
        
        # Output Layer
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_model(num_classes):
    """Create a model based on configuration."""
    if USE_TRANSFER_LEARNING:
        print("Creating transfer learning model...")
        model, base_model = create_transfer_model(num_classes)
        return model, base_model
    else:
        print("Creating simple CNN model...")
        model = create_simple_model(num_classes)
        return model, None

def train_model(model, train_generator, val_generator, num_train_samples, num_val_samples):
    """Train the model with OneCycleLR policy and other callbacks."""
    print("\n=== Starting Model Training ===")
    
    # Calculate steps per epoch
    steps_per_epoch = num_train_samples // BATCH_SIZE
    validation_steps = num_val_samples // BATCH_SIZE
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    
    # Create callbacks
    callbacks = [
        # Model checkpoint to save best model
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=VERBOSE
        ),
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=VERBOSE
        ),
        # OneCycleLR for learning rate scheduling
        OneCycleLR(
            max_lr=MAX_LR,
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1e4
        ),
        # TensorBoard for monitoring training
        tf.keras.callbacks.TensorBoard(
            log_dir='logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            histogram_freq=1
        )
    ]
    
    # Compile model with Adam optimizer
    print("\nCompiling model...")
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nStarting model training...")
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=VERBOSE
    )
    
    print("\n=== Training Completed ===")
    return history

def cross_validate(X_paths, y, num_classes, category_mapping, class_weight_dict):
    """Perform k-fold cross-validation."""
    print(f"Performing {K_FOLDS}-fold cross-validation...")
    
    # Create k-fold cross-validation
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    # Store results
    fold_scores = []
    
    # Perform k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_paths)):
        print(f"\nFold {fold+1}/{K_FOLDS}")
        
        # Split data
        X_paths_train, X_paths_val = [X_paths[i] for i in train_idx], [X_paths[i] for i in val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create model
        model, base_model = create_model(num_classes)
        
        # Train model
        history = train_model(model, X_paths_train, X_paths_val, len(X_paths_train), len(X_paths_val))
        
        # Evaluate model
        val_generator = create_data_generator(X_paths_val, y_val, BATCH_SIZE, is_training=False)
        score = model.evaluate(val_generator, steps=len(X_paths_val) // BATCH_SIZE, verbose=0)
        
        print(f"Fold {fold+1} validation accuracy: {score[1]:.4f}")
        fold_scores.append(score[1])
        
        # Unfreeze some layers for fine-tuning if using transfer learning
        if USE_TRANSFER_LEARNING and base_model is not None:
            print("Fine-tuning the model...")
            
            # Unfreeze the last few layers of the base model
            for layer in base_model.layers[-30:]:
                layer.trainable = True
            
            # Recompile the model with a lower learning rate
            model.compile(
                optimizer=Adam(learning_rate=LEARNING_RATE/10),
                loss=loss,
                metrics=['accuracy']
            )
            
            # Train the model again
            history = train_model(model, X_paths_train, X_paths_val, len(X_paths_train), len(X_paths_val))
            
            # Evaluate model again
            score = model.evaluate(val_generator, steps=len(X_paths_val) // BATCH_SIZE, verbose=0)
            
            print(f"Fold {fold+1} validation accuracy after fine-tuning: {score[1]:.4f}")
            fold_scores[fold] = score[1]
    
    # Print average score
    print(f"\nAverage validation accuracy: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    
    return fold_scores

def plot_training_history(history):
    """Plot training and validation metrics with enhanced visualization."""
    # Set style
    plt.style.use('seaborn')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # 1. Accuracy Plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history.history['accuracy'], label='Training', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, pad=15)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=12)
    
    # Add accuracy annotations
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    ax1.annotate(f'Final Train: {final_train_acc:.4f}', 
                xy=(len(history.history['accuracy'])-1, final_train_acc),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
    ax1.annotate(f'Final Val: {final_val_acc:.4f}\nBest Val: {best_val_acc:.4f}',
                xy=(len(history.history['val_accuracy'])-1, final_val_acc),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
    
    # 2. Loss Plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history.history['loss'], label='Training', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, pad=15)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=12)
    
    # Add loss annotations
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    best_val_loss = min(history.history['val_loss'])
    ax2.annotate(f'Final Train: {final_train_loss:.4f}',
                xy=(len(history.history['loss'])-1, final_train_loss),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
    ax2.annotate(f'Final Val: {final_val_loss:.4f}\nBest Val: {best_val_loss:.4f}',
                xy=(len(history.history['val_loss'])-1, final_val_loss),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
    
    # 3. Learning Rate Plot
    if 'lr' in history.history:
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(history.history['lr'], label='Learning Rate', linewidth=2, color='green')
        ax3.set_title('Learning Rate Schedule', fontsize=14, pad=15)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Learning Rate', fontsize=12)
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(fontsize=12)
        ax3.set_yscale('log')
    
    # 4. Training Progress Summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Calculate metrics
    epochs = len(history.history['accuracy'])
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    final_epoch = epochs
    
    summary_text = (
        f"Training Summary:\n\n"
        f"Total Epochs: {epochs}\n"
        f"Best Epoch: {best_epoch}\n\n"
        f"Best Validation Accuracy: {best_val_acc:.4f}\n"
        f"Final Validation Accuracy: {final_val_acc:.4f}\n"
        f"Accuracy Improvement: {(final_val_acc - history.history['val_accuracy'][0]):.4f}\n\n"
        f"Best Validation Loss: {best_val_loss:.4f}\n"
        f"Final Validation Loss: {final_val_loss:.4f}\n"
        f"Loss Improvement: {(history.history['val_loss'][0] - final_val_loss):.4f}\n"
    )
    
    ax4.text(0.1, 0.5, summary_text, fontsize=12, va='center',
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_category_mapping(category_mapping):
    """Save the category mapping to a JSON file."""
    with open(CATEGORY_MAPPING_PATH, 'w') as f:
        json.dump(category_mapping, f)

def load_category_mapping():
    """Load the category mapping from a JSON file."""
    with open(CATEGORY_MAPPING_PATH, 'r') as f:
        return json.load(f)

def predict_with_confidence(model, image_path, category_mapping):
    """Predict the waste category with confidence score."""
    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    prediction = model.predict(img, verbose=0)
    confidence = prediction[0][0]  # Confidence for recyclable class
    
    # Determine category based on confidence threshold
    if confidence >= CONFIDENCE_THRESHOLD:
        category = "Recyclable"
    elif confidence <= (1 - CONFIDENCE_THRESHOLD):
        category = "Organic"
    else:
        category = "Organic (Low Confidence)"
    
    return category, confidence

def predict_multiple_images(model, image_paths, category_mapping):
    """Predict waste categories for multiple images with confidence scores."""
    results = []
    for img_path in image_paths:
        try:
            category, confidence = predict_with_confidence(model, img_path, category_mapping)
            results.append({
                'image_path': img_path,
                'category': category,
                'confidence': float(confidence),
                'recommendation': 'Recyclable' if confidence >= CONFIDENCE_THRESHOLD else 'Organic'
            })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    return results

def main():
    """Main function to run the waste classification model."""
    try:
        print("\n=== Starting Waste Classification Model ===")
        
        # Print debugging information
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {SCRIPT_DIR}")
        print(f"Workspace directory: {WORKSPACE_DIR}")
        print(f"Data directory: {DATA_DIR}")
        print(f"Data directory exists: {os.path.exists(DATA_DIR)}")
        if os.path.exists(DATA_DIR):
            print(f"Contents of DATA_DIR: {os.listdir(DATA_DIR)}")
        
        # Load or prepare data
        print("\nLoading and preprocessing data...")
        X_paths_train, X_paths_val, y_train, y_val, num_classes, category_mapping, class_weight_dict = load_and_preprocess_data()
        
        if len(X_paths_train) == 0 or len(X_paths_val) == 0:
            raise ValueError("No training or validation data available")
        
        # Save category mapping
        save_category_mapping(category_mapping)
        
        # Create data generators
        print("\nCreating data generators...")
        train_generator, train_steps = create_data_generator(X_paths_train, y_train, BATCH_SIZE, is_training=True, class_weights=class_weight_dict)
        val_generator, val_steps = create_data_generator(X_paths_val, y_val, BATCH_SIZE, is_training=False)
        
        print(f"Training steps per epoch: {train_steps}")
        print(f"Validation steps per epoch: {val_steps}")
        
        # Perform cross-validation if enabled
        if USE_CROSS_VALIDATION:
            print("\nPerforming cross-validation...")
            # Combine training and validation data for cross-validation
            X_paths_all = X_paths_train + X_paths_val
            y_all = np.vstack((y_train, y_val))
            
            # Perform cross-validation
            fold_scores = cross_validate(X_paths_all, y_all, num_classes, category_mapping, class_weight_dict)
            
            # Use the best model for final evaluation
            best_fold = np.argmax(fold_scores)
            print(f"\nUsing model from fold {best_fold+1} for final evaluation")
            
            # Load the best model
            model = keras.models.load_model(f"model_fold_{best_fold+1}.h5")
        else:
            # Create or load model
            if os.path.exists(MODEL_SAVE_PATH):
                print(f"\nLoading existing model from {MODEL_SAVE_PATH}")
                model = keras.models.load_model(MODEL_SAVE_PATH)
                print("Model loaded successfully")
            else:
                print("\nCreating new model...")
                model, base_model = create_model(num_classes)
                print("Model created successfully")
            
            # Train the model
            history = train_model(model, train_generator, val_generator, len(X_paths_train), len(X_paths_val))
            
            # Plot training history
            plot_training_history(history)
            
            # Fine-tune the model if using transfer learning
            if USE_TRANSFER_LEARNING and base_model is not None:
                print("\nFine-tuning the model...")
                
                # Unfreeze the last few layers of the base model
                for layer in base_model.layers[-30:]:
                    layer.trainable = True
                
                # Recompile the model with a lower learning rate
                model.compile(
                    optimizer=Adam(learning_rate=LEARNING_RATE/10),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Train the model again
                history = train_model(model, train_generator, val_generator, len(X_paths_train), len(X_paths_val))
                
                # Plot training history
                plot_training_history(history)
        
        print("\n=== Training Process Completed Successfully! ===")
        
    except Exception as e:
        print(f"\nError in main: {e}")
        raise

def predict_from_saved_model(image_paths):
    """Load a saved model and make predictions."""
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_SAVE_PATH}")
    if not os.path.exists(CATEGORY_MAPPING_PATH):
        raise FileNotFoundError(f"Category mapping file not found at {CATEGORY_MAPPING_PATH}")
    
    # Load the model and category mapping
    model = keras.models.load_model(MODEL_SAVE_PATH)
    category_mapping = load_category_mapping()
    
    # Make predictions
    results = predict_multiple_images(model, image_paths, category_mapping)
    
    # Print results
    for result in results:
        print(f"\nImage: {result['image_path']}")
        print(f"Predicted category: {result['category']}")
        print(f"Confidence: {result['confidence']:.2f}")
    
    return results

# Add Mixup data augmentation
def mixup(x, y, alpha=0.2):
    """Apply Mixup data augmentation to a batch of images and labels."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = tf.shape(x)[0]
    index = tf.random.shuffle(tf.range(batch_size))
    
    mixed_x = lam * x + (1 - lam) * tf.gather(x, index)
    mixed_y = lam * y + (1 - lam) * tf.gather(y, index)
    
    return mixed_x, mixed_y

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--predict":
        # If --predict flag is provided, run prediction on provided images
        if len(sys.argv) < 3:
            print("Please provide image paths to predict")
            sys.exit(1)
        image_paths = sys.argv[2:]
        predict_from_saved_model(image_paths)
    else:
        # Otherwise, train the model
        main()
