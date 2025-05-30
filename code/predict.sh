#!/bin/bash

# Simple shell script to predict garbage type from an image

# Check if an image path is provided
if [ $# -lt 1 ]; then
    echo "Usage: ./predict.sh <image_path> [--display]"
    echo "  <image_path>: Path to the image to classify"
    echo "  --display: Optional flag to display the image with prediction results"
    exit 1
fi

# Set default values
IMAGE_PATH=""
DISPLAY_FLAG=""

# Parse arguments
for arg in "$@"; do
    if [[ $arg == --display ]]; then
        DISPLAY_FLAG="--display"
    elif [[ $arg != --* ]]; then
        IMAGE_PATH=$arg
    fi
done

# Check if image path is provided
if [ -z "$IMAGE_PATH" ]; then
    echo "Error: Image path is required"
    echo "Usage: ./predict.sh <image_path> [--display]"
    exit 1
fi

# Check if image exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: Image file not found at $IMAGE_PATH"
    exit 1
fi

# Run the prediction script
python3 simple_predict.py --image "$IMAGE_PATH" $DISPLAY_FLAG 