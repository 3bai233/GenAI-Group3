#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# Activate the correct conda environment (adjust environment name as needed)

# Change to the script directory
cd YOUR_WORKSPACE_PATH/bbox_annotation

# Define parameters
MODEL_PATH="YOUR_MODEL_PATH"
DEVICE="cuda"
OUTPUT_DIR="YOUR_OUTPUT_DIR"

# Can pass multiple JSON file paths separated by spaces
JSON_FILES="YOUR_JSON_FILES_PATH"


echo "Starting batch prediction task..."

# Execute Python script
# Remove --visualize if you don't want to save visualized images
python ui_bbox.py \
    --model_path "$MODEL_PATH" \
    --json_files $JSON_FILES \
    --device "$DEVICE" \
    --visualize \
    --output_dir "$OUTPUT_DIR"

echo "Task completed!"
