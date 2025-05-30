import os
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import cv2
import sklearn
from sklearn.model_selection import train_test_split

print("NumPy version:", np.__version__)
print("TensorFlow:", tf)
print("OpenCV version:", cv2.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("Scikit-learn version:", sklearn.__version__)

# Try to create a simple TensorFlow model
try:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(5,), activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    print("Model created successfully!")
except Exception as e:
    print("Error creating model:", e) 