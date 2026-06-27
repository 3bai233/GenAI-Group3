#!/usr/bin/env bash
set -euo pipefail

# Run Holo2 + BAMI with an API-based judge instead of the local judge model.
# Supported API_JUDGE_TYPE values:
#   openai          OpenAI-compatible APIs: OpenAI, Azure, OpenRouter, LingYaAI, etc.
#   gemini          Google Gemini standard API
#   gemini_thinking Google Gemini thinking API
#
# Examples:
#   OPENAI_API_KEY=sk-xxx bash group5GUI/scripts/run_holo2_bami_api_judge.sh 8B
#   API_JUDGE_TYPE=gemini GEMINI_API_KEY=xxx bash group5GUI/scripts/run_holo2_bami_api_judge.sh 8B
#   API_JUDGE_TYPE=openai API_KEY=xxx BASE_URL=https://api.example.com/v1 JUDGE_MODEL=gemini-3-pro-preview bash group5GUI/scripts/run_holo2_bami_api_judge.sh 8B

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${GROUP_ROOT}/.." && pwd)"
FEEDCOG_DIR="${FEEDCOG_DIR:-${REPO_ROOT}/FeedCoG}"
SCREENSPOT_ROOT="${SCREENSPOT_ROOT:-${GROUP_ROOT}/datasets/ScreenSpot-Pro}"

pick_existing_dir() {
  for candidate in "$@"; do
    if [[ -d "${candidate}" ]]; then
      echo "${candidate}"
      return
    fi
  done
  echo "$1"
}

MODEL_SIZE="${1:-${MODEL_SIZE:-8B}}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${MODEL_SIZE}" in
  4B|4b)
    MODEL_LABEL="4b"
    DEFAULT_MODEL="$(pick_existing_dir "/data/models/Holo2-4B" "/data2/datasets/Holo2-4B" "${HOME}/models/Holo2-4B")"
    DEFAULT_GPUS="0"
    ;;
  8B|8b)
    MODEL_LABEL="8b"
    DEFAULT_MODEL="$(pick_existing_dir "/data/models/Holo2-8B" "/data2/datasets/Holo2-8B" "${HOME}/models/Holo2-8B")"
    DEFAULT_GPUS="0"
    ;;
  30B|30b|30B-A3B|30b-a3b)
    MODEL_LABEL="30b"
    DEFAULT_MODEL="$(pick_existing_dir "/data/models/Holo2-30B-A3B" "/data2/datasets/Holo2-30B-A3B" "${HOME}/models/Holo2-30B-A3B")"
    DEFAULT_GPUS="0,1,2,3,4,5,6,7"
    ;;
  *)
    echo "Unsupported MODEL_SIZE: ${MODEL_SIZE}. Use 4B, 8B, or 30B."
    exit 2
    ;;
esac

GROUNDING_MODEL="${GROUNDING_MODEL:-${DEFAULT_MODEL}}"
SCREENSPOT_IMGS="${SCREENSPOT_IMGS:-${SCREENSPOT_ROOT}/images}"
SCREENSPOT_TEST="${SCREENSPOT_TEST:-${SCREENSPOT_ROOT}/annotations}"
PHYSICAL_GPUS="${PHYSICAL_GPUS:-${CUDA_GPUS:-${DEFAULT_GPUS}}}"

TASK="${TASK:-all}"
LANGUAGE="${LANGUAGE:-en}"
GT_TYPE="${GT_TYPE:-positive}"
INST_STYLE="${INST_STYLE:-instruction}"
PROMPT_TYPE="${PROMPT_TYPE:-holo2_json}"
ROOT_PATH="${ROOT_PATH:-eval_results}"
API_JUDGE_TYPE="${API_JUDGE_TYPE:-openai}"
BASE_URL="${BASE_URL:-${OPENAI_BASE_URL:-None}}"
THINKING_BUDGET="${THINKING_BUDGET:-8192}"

case "${API_JUDGE_TYPE}" in
  openai)
    API_KEY_VALUE="${API_KEY:-${OPENAI_API_KEY:-${OPENROUTER_API_KEY:-}}}"
    JUDGE_MODEL="${JUDGE_MODEL:-${MODEL:-gpt-4o}}"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-${API_KEY_VALUE}}"
    EXTRA_API_ARGS=(--judge_api_type openai --gpt_base_url "${BASE_URL}" --gpt_model "${JUDGE_MODEL}")
    ;;
  gemini)
    API_KEY_VALUE="${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}"
    JUDGE_MODEL="${JUDGE_MODEL:-${GEMINI_MODEL:-gemini-2.0-flash}}"
    export GEMINI_API_KEY="${GEMINI_API_KEY:-${API_KEY_VALUE}}"
    EXTRA_API_ARGS=(--judge_api_type gemini --gpt_model "${JUDGE_MODEL}")
    ;;
  gemini_thinking)
    API_KEY_VALUE="${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}"
    JUDGE_MODEL="${JUDGE_MODEL:-${GEMINI_MODEL:-gemini-2.5-pro-preview-05-06}}"
    export GEMINI_API_KEY="${GEMINI_API_KEY:-${API_KEY_VALUE}}"
    EXTRA_API_ARGS=(--judge_api_type gemini_thinking --gpt_model "${JUDGE_MODEL}" --thinking_budget "${THINKING_BUDGET}")
    ;;
  *)
    echo "Unsupported API_JUDGE_TYPE: ${API_JUDGE_TYPE}. Use openai, gemini, or gemini_thinking."
    exit 2
    ;;
esac

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_NAME="${LOG_NAME:-group5-holo2-${MODEL_LABEL}-bami-api-${API_JUDGE_TYPE}-${TIMESTAMP}}"

if [[ ! -f "${FEEDCOG_DIR}/lenovo_eval_ss_pro.py" ]]; then
  echo "Error: lenovo_eval_ss_pro.py not found under FEEDCOG_DIR=${FEEDCOG_DIR}"
  exit 1
fi

for required_dir in "${GROUNDING_MODEL}" "${SCREENSPOT_IMGS}" "${SCREENSPOT_TEST}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "Error: required directory not found: ${required_dir}"
    echo "Set the corresponding environment variable before running this script."
    exit 1
  fi
done

if [[ -z "${API_KEY_VALUE}" ]]; then
  echo "Error: API key is not set for API_JUDGE_TYPE=${API_JUDGE_TYPE}."
  echo "For openai, set API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY."
  echo "For gemini/gemini_thinking, set GEMINI_API_KEY or GOOGLE_API_KEY."
  exit 1
fi

echo "========================================================================"
echo "Group5 GUI: Holo2-${MODEL_SIZE} + BAMI + API Judge"
echo "========================================================================"
echo "FeedCoG dir:      ${FEEDCOG_DIR}"
echo "Grounding model:  ${GROUNDING_MODEL}"
echo "Images:           ${SCREENSPOT_IMGS}"
echo "Annotations:      ${SCREENSPOT_TEST}"
echo "Task:             ${TASK}"
echo "Prompt:           ${PROMPT_TYPE}"
echo "GPUs:             ${PHYSICAL_GPUS}"
echo "Judge type:       ${API_JUDGE_TYPE}"
echo "Judge model:      ${JUDGE_MODEL}"
echo "Judge base URL:   ${BASE_URL}"
echo "Log name:         ${LOG_NAME}"
echo "Extra args:       $*"
echo "========================================================================"

cd "${FEEDCOG_DIR}"
mkdir -p "${ROOT_PATH}"

CUDA_VISIBLE_DEVICES="${PHYSICAL_GPUS}" TOKENIZERS_PARALLELISM=false \
python lenovo_eval_ss_pro.py \
  --model_path "${GROUNDING_MODEL}" \
  --screenspot_imgs "${SCREENSPOT_IMGS}" \
  --screenspot_test "${SCREENSPOT_TEST}" \
  --task "${TASK}" \
  --language "${LANGUAGE}" \
  --gt_type "${GT_TYPE}" \
  --inst_style "${INST_STYLE}" \
  --prompt_type "${PROMPT_TYPE}" \
  --use_reground_judge_two_images \
  --use_api_judge \
  "${EXTRA_API_ARGS[@]}" \
  --root_path "${ROOT_PATH}" \
  --log_name "${LOG_NAME}" \
  "$@" \
  2>&1 | tee "${ROOT_PATH}/${LOG_NAME}.log"

echo "========================================================================"
echo "Done."
echo "Result JSON: ${FEEDCOG_DIR}/${ROOT_PATH}/${LOG_NAME}.json"
echo "Run log:     ${FEEDCOG_DIR}/${ROOT_PATH}/${LOG_NAME}.log"
echo "========================================================================"

if [[ -f "${FEEDCOG_DIR}/analyze_bbox2_contribution.py" ]]; then
  python analyze_bbox2_contribution.py "${ROOT_PATH}/${LOG_NAME}.json" || true
fi
