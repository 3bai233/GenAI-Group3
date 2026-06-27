# EgoXR-GUI Evaluation Suite

This directory encompasses the testing framework for the EgoXR-GUI benchmark. EgoXR-GUI systematically evaluates Multimodal Large Language Models (MLLMs) and GUI agents on their interface grounding capabilities within extended reality across three tasks: Direct, Spatial, and Semantic Grounding.

Dataset can be downloaded from: https://huggingface.co/datasets/Anonymous114/EgoXR-GUI/

## Overview

- `eval_egoxr_gui.py`: The core script used to launch the evaluation routines against the EgoXR-GUI dataset.
- `run_models.sh`: A shell wrapper that automates the sequential or iterative execution of various built-in foundation models.
- `eval_finetune.py` / `run_finetune_test.sh`: Specific scripts dedicated to evaluating custom post-SFT enhanced models to validate performance gains on Semantic Grounding scenarios.
- `run_task_router.sh`: Deploys the multi-model agentic reasoning approach (Task Router Strategy) to selectively inject higher-order reasoning over complex user instructions while falling back to direct localization for simpler queries.
- `visual_data.py`: Utilities tailored for parsing, formatting, and visualizing the evaluation dataset annotations and outputs.
- `models/`: Interface standardizations and execution logic for major MLLMs to interface directly with the evaluation runner (e.g. `qwen3vl.py`, `ui_venus1_5.py`). It also hosts `gui_agents/`, which builds upon the Agent-S framework to instantiate our Task Router configurations.