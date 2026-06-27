#!/usr/bin/env bash
set -e

MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-10}"

cmd=(
    python -m xr_gui_grounding
    --input YOUR_INPUT_PATH
    --output ./outputs/part_001
    --vlm-model qwen3.6-plus
    --env .env
    --num-annotations 3
    --max-images 100
    --bilingual
)

attempt=1
while true; do
    echo "[part_001] Run ${attempt}/${MAX_RETRIES}"
    if "${cmd[@]}"; then
        echo "[part_001] Completed successfully."
        break
    fi

    exit_code=$?
    if [ "${attempt}" -ge "${MAX_RETRIES}" ]; then
        echo "[part_001] Failed after ${attempt} attempts (exit code ${exit_code})."
        exit "${exit_code}"
    fi

    echo "[part_001] Failed with exit code ${exit_code}. Retrying in ${RETRY_SLEEP_SECONDS} seconds..."
    sleep "${RETRY_SLEEP_SECONDS}"
    attempt=$((attempt + 1))
done