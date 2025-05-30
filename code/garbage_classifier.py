import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import json 
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Define constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16  # Reduced batch size for better stability
EPOCHS = 20  # Increased epochs
TRAIN_DIR = 'DATASET/TRAIN'  # Fixed path
TEST_DIR = 'DATASET/TEST'  # Fixed path
CATEGORY_MAPPING = {"R": "Recyclable", "O": "Organic"}
NUM_CLASSES = 2

def load_category_mapping():
    """Load the category mapping from JSON file"""
    try:
        with open('category_mapping.json', 'r') as f:
            mapping = json.load(f)
        return mapping
    except FileNotFoundError:
        print("Category mapping file not found. Using default mapping.")
        return {"R": 0, "O": 1}

def create_data_generators():
    """Create data generators for training and validation"""
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,  # Increased rotation range
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,  # Added vertical flip 
        brightness_range=[0.8, 1.2],  # Added brightness adjustment
        fill_mode='nearest',
        validation_split=0.2
    )

    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # Training generator
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    # Validation generator
    validation_generator = val_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Test generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator

def create_model():
    """Create a transfer learning model based on MobileNetV2"""
    # Load the pre-trained model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)  # Added batch normalization
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)  # Added another dense layer
    x = BatchNormalization()(x)  # Added batch normalization
    x = Dropout(0.3)(x)  # Reduced dropout
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def train_model(model, train_generator, validation_generator):
    """Train the model with callbacks"""
    # Create callbacks
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        'models/garbage_classifier_best.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    # Compile the model with legacy Adam optimizer for better performance on M1/M2 Macs
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[reduce_lr, early_stopping, model_checkpoint]
    )
    
    return history

def fine_tune_model(model, train_generator, validation_generator):
    """Fine-tune the model by unfreezing some layers"""
    # Unfreeze the top layers of the base model
    for layer in model.layers[-30:]:  # Unfreeze more layers
        layer.trainable = True
    
    # Recompile the model with legacy Adam optimizer
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.00001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create callbacks
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    # Fine-tune the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[reduce_lr, early_stopping]
    )
    
    return history

def plot_training_history(history):
    """Plot the training history"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('logs/training_history.png')
    plt.close()

def evaluate_model(model, test_generator):
    """Evaluate the model on the test set"""
    # Get predictions
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Get class names
    class_names = list(test_generator.class_indices.keys())
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('logs/confusion_matrix.png')
    plt.close()
    
    return predictions

def save_model(model):
    """Save the trained model"""
    model.save('models/garbage_classifier.h5')  # Fixed path
    print("Model saved to 'models/garbage_classifier.h5'")

def predict_image(model, image_path):
    """Predict the class of a single image"""
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # Get class name
    class_names = list(CATEGORY_MAPPING.values())
    predicted_class_name = class_names[predicted_class]
    
    return predicted_class_name, confidence

def main():
    # Create necessary directories
    os.makedirs('models', exist_ok=True)  # Fixed path
    os.makedirs('logs', exist_ok=True)  # Fixed path
    
    # Load category mapping
    category_mapping = load_category_mapping()
    
    # Create data generators
    train_generator, validation_generator, test_generator = create_data_generators()
    
    # Create and train the model
    model = create_model()
    history = train_model(model, train_generator, validation_generator)
    
    # Fine-tune the model
    fine_tune_history = fine_tune_model(model, train_generator, validation_generator)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model
    predictions = evaluate_model(model, test_generator)
    
    # Save the model
    save_model(model)
    
    # Example prediction
    print("\nExample prediction:")
    test_image_path = os.path.join(TEST_DIR, 'O', os.listdir(os.path.join(TEST_DIR, 'O'))[0])
    predicted_class, confidence = predict_image(model, test_image_path)
    print(f"Image: {test_image_path}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main() 