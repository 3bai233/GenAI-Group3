# GenAI-Group3

Group 3 course project core code for GUI grounding and evaluation.

This work references the BAMI project implementation.

Author: zhangbo <226653803@qq.com>

## Structure

- `FeedCoG/`: evaluation pipeline and judging utilities.
- `group3GUI/scripts/`: environment check and run scripts.
- `group3GUI/requirements-holo-bami.txt`: Python dependencies.

## Usage

```bash
pip install -r group3GUI/requirements-holo-bami.txt
bash group3GUI/scripts/check_env.sh
bash group3GUI/scripts/run_holo2_bami.sh 8B
```

Prepare the required models and dataset locally, then configure paths with environment variables such as `GROUNDING_MODEL`, `LOCAL_JUDGE_MODEL`, and `SCREENSPOT_ROOT`.
