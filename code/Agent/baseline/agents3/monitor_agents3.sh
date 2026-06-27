#!/usr/bin/env bash
# Monitor and auto-restart UI-TARS vllm and AgentS3 run.
# Usage: nohup bash scripts/bash/monitor_agents3.sh > logs/monitor.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/../.."

VLLM_PORT=8000
VLLM_READY_TIMEOUT=300   # seconds to wait for vllm to come up
CHECK_INTERVAL=60         # seconds between health checks

VLLM_CMD=(
  env CUDA_VISIBLE_DEVICES=6,7 VLLM_WORKER_MULTIPROC_METHOD=spawn
  ./ui_tars_venv/bin/vllm serve UI-TARS-1.5-7B
  --host 127.0.0.1
  --port "$VLLM_PORT"
  --served-model-name ByteDance-Seed/UI-TARS-1.5-7B
  --max-model-len 32768
  --tensor-parallel-size 2
)

AGENTS3_CMD=(
  env NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
  OSWORLD_DOCKER_IMAGE=happysixd/osworld-docker
  ./ui_tars_venv/bin/python scripts/python/run_agents3.py
  --provider_name docker
  --path_to_vm docker_vm_data/Ubuntu.qcow2
  --headless
  --observation_type screenshot
  --action_space pyautogui
  --num_envs 8
  --model gpt-4o
  --model_provider openai
  --model_url "https://www.dmxapi.cn/v1"
  --model_api_key "<REDACTED_API_KEY>"
  --ground_provider openai
  --ground_url "http://localhost:${VLLM_PORT}/v1"
  --ground_api_key EMPTY
  --ground_model ByteDance-Seed/UI-TARS-1.5-7B
  --grounding_width 1920
  --grounding_height 1080
  --max_steps 50
  --test_all_meta_path evaluation_examples/test_nogdrive.json
  --domain all
  --result_dir ./results_agents3
  --sleep_after_execution 3
)

VLLM_PID=""
AGENTS3_PID=""

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

is_alive() { [ -n "$1" ] && kill -0 "$1" 2>/dev/null; }

vllm_healthy() {
  curl -sf --max-time 5 --noproxy '*' "http://127.0.0.1:${VLLM_PORT}/v1/models" > /dev/null 2>&1
}

wait_vllm_ready() {
  log "Waiting up to ${VLLM_READY_TIMEOUT}s for vllm to become ready..."
  local elapsed=0
  while [ "$elapsed" -lt "$VLLM_READY_TIMEOUT" ]; do
    if vllm_healthy; then
      log "vllm is ready."
      return 0
    fi
    sleep 10
    elapsed=$((elapsed + 10))
  done
  log "ERROR: vllm did not become ready within ${VLLM_READY_TIMEOUT}s."
  return 1
}

start_vllm() {
  log "Starting UI-TARS vllm server..."
  mkdir -p logs
  local logfile="logs/vllm_serve_$(date +%Y%m%d_%H%M%S).log"
  "${VLLM_CMD[@]}" > "$logfile" 2>&1 &
  VLLM_PID=$!
  log "vllm started with PID $VLLM_PID, log: $logfile"
}

clean_incomplete_tasks() {
  local removed=0
  for dir in $(find results_agents3 -mindepth 5 -maxdepth 5 -type d 2>/dev/null); do
    if [ ! -f "$dir/result.txt" ]; then
      rm -rf "$dir"
      removed=$((removed + 1))
    fi
  done
  [ "$removed" -gt 0 ] && log "Cleaned $removed incomplete task dirs."
}

start_agents3() {
  clean_incomplete_tasks
  local completed
  completed=$(find results_agents3 -name "result.txt" 2>/dev/null | wc -l)
  log "Starting AgentS3 ($completed tasks done so far)..."
  mkdir -p logs
  local logfile="logs/run_agents3_$(date +%Y%m%d_%H%M%S).log"
  "${AGENTS3_CMD[@]}" > "$logfile" 2>&1 &
  AGENTS3_PID=$!
  log "AgentS3 started with PID $AGENTS3_PID, log: $logfile"
}

# ── Initial startup ──────────────────────────────────────────────────────────
log "Monitor started (PID $$). Check interval: ${CHECK_INTERVAL}s."

if vllm_healthy; then
  log "vllm already running on port $VLLM_PORT."
  # Find its PID
  VLLM_PID=$(pgrep -f "vllm.*port.*${VLLM_PORT}" | head -1 || true)
else
  start_vllm
  wait_vllm_ready || { log "FATAL: cannot start vllm. Exiting."; exit 1; }
fi

# Find existing AgentS3 process if any
AGENTS3_PID=$(pgrep -f "run_agents3.py" | head -1 || true)
if is_alive "$AGENTS3_PID"; then
  log "AgentS3 already running with PID $AGENTS3_PID."
else
  start_agents3
fi

# ── Watch loop ───────────────────────────────────────────────────────────────
while true; do
  sleep "$CHECK_INTERVAL"

  # 1. Check vllm
  if ! vllm_healthy; then
    log "WARNING: vllm health check failed."
    if is_alive "$VLLM_PID"; then
      log "vllm process $VLLM_PID still alive but not responding — killing it."
      kill "$VLLM_PID" 2>/dev/null
      sleep 5
    fi
    # Also kill AgentS3 so it doesn't generate bad results without grounding
    if is_alive "$AGENTS3_PID"; then
      log "Stopping AgentS3 (PID $AGENTS3_PID) until vllm recovers."
      kill "$AGENTS3_PID" 2>/dev/null
      AGENTS3_PID=""
    fi
    start_vllm
    wait_vllm_ready || { log "ERROR: vllm restart failed. Will retry next cycle."; continue; }
  fi

  # 2. Check AgentS3
  if ! is_alive "$AGENTS3_PID"; then
    completed=$(find results_agents3 -name "result.txt" 2>/dev/null | wc -l)
    total=369
    if [ "$completed" -ge "$total" ]; then
      log "All $total tasks complete. Monitor exiting."
      exit 0
    fi
    log "AgentS3 not running ($completed/$total tasks done). Restarting..."
    start_agents3
  fi

  # 3. Status heartbeat
  completed=$(find results_agents3 -name "result.txt" 2>/dev/null | wc -l)
  avg=$(find results_agents3 -name "result.txt" 2>/dev/null | xargs cat 2>/dev/null \
        | awk '{s+=$1;c++} END{if(c>0) printf "%.1f%%", s/c*100; else print "n/a"}')
  log "Status: $completed tasks done, avg score $avg | vllm PID=${VLLM_PID} agents3 PID=${AGENTS3_PID}"
done
