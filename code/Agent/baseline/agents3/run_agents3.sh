#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

mkdir -p logs

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-logs/run_agents3_${TIMESTAMP}.log}"

echo "Logging to ${LOG_FILE}"

./ui_tars_venv/bin/python scripts/python/run_agents3.py \
  --provider_name docker \
  --headless \
  --observation_type screenshot \
  --action_space pyautogui \
  --model "${MODEL:-gpt-4o}" \
  --model_provider openai \
  --model_url "https://www.dmxapi.cn/v1" \
  --model_api_key "${OPENAI_API_KEY:-<REDACTED_API_KEY>}" \
  --ground_provider openai \
  --ground_url "http://localhost:8000/v1" \
  --ground_api_key "EMPTY" \
  --ground_model "/share/home/group3/agent/OSWorld/UI-TARS-1.5-7B" \
  --grounding_width 1920 \
  --grounding_height 1080 \
  --max_steps "${MAX_STEPS:-50}" \
  --num_envs "${NUM_ENVS:-1}" \
  --test_all_meta_path "${TEST_META_PATH:-evaluation_examples/test_nogdrive.json}" \
  --domain "${DOMAIN:-all}" \
  --result_dir "${RESULT_DIR:-./results_agents3}" \
  --sleep_after_execution "${SLEEP_AFTER_EXECUTION:-3}" \
  2>&1 | tee "${LOG_FILE}"
