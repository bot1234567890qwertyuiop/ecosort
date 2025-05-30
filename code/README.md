# Garbage Classification System

This system uses deep learning to classify garbage into two categories: Organic and Recyclable. It provides confidence scores for each prediction.

## Dataset

The system uses a dataset of 22,500 images of organic and recyclable objects, divided into:
- Training data: 22,564 images (85%)
- Test data: 2,513 images (15%)

The dataset is organized as follows:
- `DATASET/TRAIN/O/`: Organic waste training images
- `DATASET/TRAIN/R/`: Recyclable waste training images
- `DATASET/TEST/O/`: Organic waste test images
- `DATASET/TEST/R/`: Recyclable waste test images

## Files

- `garbage_classifier.py`: Main script to train the model
- `predict_garbage.py`: Script to make predictions on new images
- `simple_predict.py`: Simplified script for making predictions
- `predict.sh`: Shell script for easy prediction
- `diagnose_training.py`: Script to diagnose training issues
- `category_mapping.json`: Maps class labels to indices

## Requirements

The system requires the following Python packages:
- tensorflow==2.15.0
- numpy==1.23.5
- matplotlib==3.7.1
- scikit-learn==1.2.2
- opencv-python==4.7.0.72
- Pillow==9.5.0
- tqdm==4.65.0
- seaborn==0.12.2

## Training the Model

To train the model, run:

```bash
python3 garbage_classifier.py
```

This will:
1. Load and preprocess the dataset
2. Create a transfer learning model based on MobileNetV2
3. Train the model
4. Fine-tune the model
5. Evaluate the model on the test set
6. Save the trained model to `../models/garbage_classifier.h5`
7. Generate training history plots in `../logs/`

### Addressing Increasing Loss

If you encounter increasing loss during training, the system includes several improvements to address this issue:

1. **Learning Rate Scheduling**: The model uses `ReduceLROnPlateau` to automatically reduce the learning rate when the validation loss stops improving.

2. **Early Stopping**: The model uses `EarlyStopping` to stop training when the validation loss starts to increase, preventing overfitting.

3. **Model Checkpointing**: The best model (based on validation loss) is saved during training.

4. **Improved Architecture**: The model includes batch normalization layers and an additional dense layer for better feature extraction.

5. **Enhanced Data Augmentation**: The training data is augmented with vertical flips and brightness adjustments to increase diversity.

6. **Diagnostic Tools**: Use the `diagnose_training.py` script to identify the specific cause of increasing loss:

```bash
python3 diagnose_training.py
```

This will:
- Analyze the dataset for issues like class imbalance or corrupted images
- Test different learning rates to find the optimal one
- Test different batch sizes to find the optimal one
- Test different model architectures to find the optimal one
- Provide recommendations to address the increasing loss

## Making Predictions

### Using the Python Script

To classify a new image, use the `predict_garbage.py` script:

```bash
python3 predict_garbage.py --image path/to/your/image.jpg --display
```

Or use the simplified script:

```bash
python3 simple_predict.py --image path/to/your/image.jpg --display
```

Options:
- `--model`: Path to the trained model (default: `../models/garbage_classifier.h5`)
- `--image`: Path to the image to classify (required)
- `--display`: Display the image with prediction results

### Using the Shell Script

For even easier use, you can use the shell script:

```bash
./predict.sh path/to/your/image.jpg --display
```

Options:
- `path/to/your/image.jpg`: Path to the image to classify (required)
- `--display`: Optional flag to display the image with prediction results

Example output:
```
Prediction Results:
Image: path/to/your/image.jpg
Predicted class: Recyclable
Confidence: 0.95

Confidence by class:
Recyclable: 0.95
Organic: 0.05
```

## Model Architecture

The model uses transfer learning with MobileNetV2 as the base model:
1. Pre-trained MobileNetV2 (frozen)
2. Global Average Pooling
3. Dense layer with 512 units and ReLU activation
4. Batch Normalization
5. Dropout layer (0.5)
6. Dense layer with 256 units and ReLU activation
7. Batch Normalization
8. Dropout layer (0.3)
9. Output layer with 2 units and softmax activation

## Performance

The model is evaluated using:
- Classification report (precision, recall, F1-score)
- Confusion matrix
- Training and validation accuracy/loss plots

## Updates

0.1.0
+ Database
- Used 30 category mappings for multiple garbage type recongnition

0.1.1
- Created predict file

0.2.1
+ Database
- Revamped category mappings to recyclable and organic
- Reworked the predict file
-> can graph confidence

+ LITE.2.1
- less storage needed for predictions
- comes with .jpg predict
-> can graph confidence

0.2.2
- Added .heic transformer file predicting for iPhone pictures
-> can graph confidence

0.2.3
- Added automatic .heic predict file

0.2.4
- Added batch predict for .heic files
-> can graph multiple predictions

- added different models - best and recent (.h5)

0.2.5
- Finalized prediction file
-> can graph all predictions
-> can predict all images

- Added deep analysis
-> uses gemini to correct low confidence predictions and add them into
   training setup
-> added queue and processed (o/r) folders for analysis

## Currently working on

? optimized prediction value to override to landfill type
? adapting to raspberry pi / chromebook w/ motor controls
? adding generalist model feedback for low confidence predictions
? web access
? updating LITE version -> .heic compatibility and better graphing
? adapting LITE version to raspberry pi/ chromebook w/ motor controls
? adding ULTRALITE version for speed w/ only .jpg predict/one model 
  w/ simplified verison of predict w/o model feedback(?) + much less
  memory needed
? get camera from -> chromebook/other
? clean up multiple prediction files

EcoSort - Peter Yan, 2025