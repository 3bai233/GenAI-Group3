#!/bin/bash

MODEL_PATH="YOUR_MODEL_PATH"
SCREENSPOT_IMGS="YOUR_SCREENSPOT_IMGS_PATH"
SCREENSPOT_TEST="YOUR_SCREENSPOT_TEST_PATH"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_PATH="./logs/qwen3vl-8b-finetune_$(echo ${TASK_TYPE} | tr ' ' '_')_${TIMESTAMP}.json"

mkdir -p ./logs

echo "========================================"
echo "Model:     ${MODEL_PATH}"
echo "Task type: ${TASK_TYPE}"
echo "Log:       ${LOG_PATH}"
echo "========================================"

CUDA_VISIBLE_DEVICES=0 python eval_finetune.py \
    --model_type qwen3vl_sft \
    --model_name_or_path ${MODEL_PATH} \
    --screenspot_imgs ${SCREENSPOT_IMGS} \
    --screenspot_test ${SCREENSPOT_TEST} \
    --task all \
    --inst_style instruction \
    --language en \
    --gt_type positive \
    --log_path ${LOG_PATH} \
    --max_pixels 7000000 \