#!/usr/bin/env bash
set -euo pipefail

# Run Holo2 + BAMI on ScreenSpot-Pro through FeedCoG/lenovo_eval_ss_pro.py.
# Usage:
#   bash group3GUI/scripts/run_holo2_bami.sh 8B
#   MODEL_SIZE=30B PHYSICAL_GPUS=0,1,2,3,4,5,6,7 bash group3GUI/scripts/run_holo2_bami.sh

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
    DEFAULT_GPUS="0,1"
    DEFAULT_JUDGE_GPU="1"
    ;;
  8B|8b)
    MODEL_LABEL="8b"
    DEFAULT_MODEL="$(pick_existing_dir "/data/models/Holo2-8B" "/data2/datasets/Holo2-8B" "${HOME}/models/Holo2-8B")"
    DEFAULT_GPUS="0,1"
    DEFAULT_JUDGE_GPU="1"
    ;;
  30B|30b|30B-A3B|30b-a3b)
    MODEL_LABEL="30b"
    DEFAULT_MODEL="$(pick_existing_dir "/data/models/Holo2-30B-A3B" "/data2/datasets/Holo2-30B-A3B" "${HOME}/models/Holo2-30B-A3B")"
    DEFAULT_GPUS="0,1,2,3,4,5,6,7"
    DEFAULT_JUDGE_GPU="7"
    ;;
  *)
    echo "Unsupported MODEL_SIZE: ${MODEL_SIZE}. Use 4B, 8B, or 30B."
    exit 2
    ;;
esac

GROUNDING_MODEL="${GROUNDING_MODEL:-${DEFAULT_MODEL}}"
LOCAL_JUDGE_MODEL="${LOCAL_JUDGE_MODEL:-$(pick_existing_dir "/data/models/two_box_judge_qwen3_8b/checkpoints/checkpoint-2712" "${HOME}/models/two_box_judge_qwen3_8b/checkpoints/checkpoint-2712" "/data1/model_checkpoint_GUI/two_box_judge_qwen3_8b/checkpoints/checkpoint-2712")}"
export LOCAL_JUDGE_BASE_MODEL="${LOCAL_JUDGE_BASE_MODEL:-$(pick_existing_dir "/data/models/Qwen/Qwen3-VL-8B-Instruct" "${HOME}/models/Qwen/Qwen3-VL-8B-Instruct" "Qwen/Qwen3-VL-8B-Instruct")}"
SCREENSPOT_IMGS="${SCREENSPOT_IMGS:-${SCREENSPOT_ROOT}/images}"
SCREENSPOT_TEST="${SCREENSPOT_TEST:-${SCREENSPOT_ROOT}/annotations}"
PHYSICAL_GPUS="${PHYSICAL_GPUS:-${DEFAULT_GPUS}}"
export LOCAL_JUDGE_GPU="${LOCAL_JUDGE_GPU:-${DEFAULT_JUDGE_GPU}}"

TASK="${TASK:-all}"
LANGUAGE="${LANGUAGE:-en}"
GT_TYPE="${GT_TYPE:-positive}"
INST_STYLE="${INST_STYLE:-instruction}"
PROMPT_TYPE="${PROMPT_TYPE:-holo2_json}"
ROOT_PATH="${ROOT_PATH:-eval_results}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_NAME="${LOG_NAME:-group3-holo2-${MODEL_LABEL}-bami-${TIMESTAMP}}"

if [[ ! -f "${FEEDCOG_DIR}/lenovo_eval_ss_pro.py" ]]; then
  echo "Error: lenovo_eval_ss_pro.py not found under FEEDCOG_DIR=${FEEDCOG_DIR}"
  exit 1
fi

for required_dir in "${GROUNDING_MODEL}" "${LOCAL_JUDGE_MODEL}" "${SCREENSPOT_IMGS}" "${SCREENSPOT_TEST}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "Error: required directory not found: ${required_dir}"
    echo "Set the corresponding environment variable before running this script."
    exit 1
  fi
done

echo "========================================================================"
echo "Group3 GUI: Holo2-${MODEL_SIZE} + BAMI"
echo "========================================================================"
echo "FeedCoG dir:      ${FEEDCOG_DIR}"
echo "Grounding model:  ${GROUNDING_MODEL}"
echo "Judge model:      ${LOCAL_JUDGE_MODEL}"
echo "Judge base model: ${LOCAL_JUDGE_BASE_MODEL}"
echo "Images:           ${SCREENSPOT_IMGS}"
echo "Annotations:      ${SCREENSPOT_TEST}"
echo "Task:             ${TASK}"
echo "Prompt:           ${PROMPT_TYPE}"
echo "GPUs:             ${PHYSICAL_GPUS}; judge logical cuda:${LOCAL_JUDGE_GPU}"
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
  --use_local_judge \
  --local_model_path "${LOCAL_JUDGE_MODEL}" \
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
