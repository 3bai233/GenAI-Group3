#!/bin/bash

export OPENAI_API_KEY=YOUR_API_KEY
export OPENAI_BASE_URL=YOUR_API_BASE_URL
export vLLM_API_KEY=EMPTY

python eval_screenspot_pro.py \
    --model_type agent_s \
    --provider openai \
    --model doubao-seed-2-0-pro-260215 \
    --model_url YOUR_API_BASE_URL \
    --model_api_key YOUR_API_KEY \
    --ground_provider vllm \
    --ground_model MAI-UI-8B \
    --ground_url http://localhost:8002/v1 \
    --grounding_width 1000 \
    --grounding_height 1000 \
    --screenspot_imgs YOUR_SCREENSPOT_IMGS_PATH \
    --screenspot_test YOUR_SCREENSPOT_TEST_PATH \
    --task annotation \
    --inst_style instruction \
    --language en \
    --gt_type positive \
    --log_path ./results/agents/agent_s3_router.json \
    --max_pixels 8294400 \
    --router_provider openai \
    --router_model doubao-seed-2-0-pro-260215 \
    --router_url YOUR_API_BASE_URL \
    --router_api_key YOUR_API_KEY
    