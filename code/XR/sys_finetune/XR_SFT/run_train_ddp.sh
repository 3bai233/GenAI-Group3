#!/bin/bash

# ============================================
# Qwen3-VL-8B LoRA DDP Multi-GPU Training Launcher
# RTX 4090D x2
# ============================================

# NCCL environment variables
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# CUDA settings
export CUDA_VISIBLE_DEVICES=0,1

# HuggingFace cache
export HF_HOME=YOUR_HF_CACHE_PATH
export TRANSFORMERS_CACHE=YOUR_TRANSFORMERS_CACHE_PATH

# swanlab API (if needed)
# export SWAN_LAB=your_api_key_here
# Model and output paths
MODEL_DIR="YOUR_MODEL_DIR"
OUTPUT_DIR="YOUR_OUTPUT_DIR"

# Training parameters
NUM_GPUS=2
CONFIG_FILE="config/train_config.json"

echo "=========================================="
echo "Qwen3-VL-8B LoRA DDP Training"
echo "=========================================="
echo "GPU count: $NUM_GPUS"
echo "Model path: $MODEL_DIR"
echo "Output path: $OUTPUT_DIR"
echo "Config file: $CONFIG_FILE"
echo "=========================================="

# Launch DDP training with torchrun
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29511 \
    --nnodes=1 \
    --node_rank=0 \
    train.py \
    --config $CONFIG_FILE

echo "Training task completed"
