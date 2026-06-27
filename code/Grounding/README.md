# Grounding

BAMI Grounding code for GUI grounding and ScreenSpot-Pro evaluation.

## Structure

- `FeedCoG/`: evaluation pipeline and judging utilities.
- `group3GUI/scripts/`: environment checks and run scripts.
- `group3GUI/requirements-holo-bami.txt`: Python dependencies.

## Usage

```bash
cd code/Grounding
pip install -r group3GUI/requirements-holo-bami.txt
bash group3GUI/scripts/check_env.sh
bash group3GUI/scripts/run_holo2_bami.sh 8B
```

Prepare required models and datasets locally. Configure paths with environment variables such as `GROUNDING_MODEL`, `LOCAL_JUDGE_MODEL`, `LOCAL_JUDGE_BASE_MODEL`, and `SCREENSPOT_ROOT`.

