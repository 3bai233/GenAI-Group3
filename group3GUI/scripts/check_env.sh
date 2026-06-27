#!/usr/bin/env bash
set -euo pipefail

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

GROUNDING_MODEL="${GROUNDING_MODEL:-$(pick_existing_dir "/data/models/Holo2-8B" "/data2/datasets/Holo2-8B" "${HOME}/models/Holo2-8B")}"
LOCAL_JUDGE_MODEL="${LOCAL_JUDGE_MODEL:-$(pick_existing_dir "/data/models/two_box_judge_qwen3_8b/checkpoints/checkpoint-2712" "${HOME}/models/two_box_judge_qwen3_8b/checkpoints/checkpoint-2712" "/data1/model_checkpoint_GUI/two_box_judge_qwen3_8b/checkpoints/checkpoint-2712")}"
LOCAL_JUDGE_BASE_MODEL="${LOCAL_JUDGE_BASE_MODEL:-$(pick_existing_dir "/data/models/Qwen/Qwen3-VL-8B-Instruct" "${HOME}/models/Qwen/Qwen3-VL-8B-Instruct" "Qwen/Qwen3-VL-8B-Instruct")}"
SCREENSPOT_IMGS="${SCREENSPOT_IMGS:-${SCREENSPOT_ROOT}/images}"
SCREENSPOT_TEST="${SCREENSPOT_TEST:-${SCREENSPOT_ROOT}/annotations}"

echo "== Path checks =="
for path in \
  "${FEEDCOG_DIR}" \
  "${FEEDCOG_DIR}/lenovo_eval_ss_pro.py" \
  "${GROUNDING_MODEL}" \
  "${LOCAL_JUDGE_MODEL}" \
  "${LOCAL_JUDGE_BASE_MODEL}" \
  "${SCREENSPOT_IMGS}" \
  "${SCREENSPOT_TEST}"; do
  if [[ -e "${path}" ]]; then
    echo "[OK]   ${path}"
  else
    echo "[MISS] ${path}"
  fi
done

echo
echo "== Python package checks =="
python - <<'PY'
import importlib

packages = [
    "torch",
    "transformers",
    "PIL",
    "tqdm",
    "accelerate",
    "qwen_vl_utils",
]

for name in packages:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK]   {name}: {version}")
    except Exception as exc:
        print(f"[MISS] {name}: {exc}")
PY

echo
echo "== GPU checks =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
else
  echo "[MISS] nvidia-smi not found"
fi
