# GenAI-Group3

Group 3 course project repository for GUI grounding, GUI agents, and XR GUI grounding.

## Overview

This repository contains three related parts:

- `Grounding`: BAMI-based GUI grounding code and ScreenSpot-Pro style evaluation utilities.
- `Agent`: OSWorld GUI agent baselines and BAMI-enhanced agent variants.
- `XR`: EgoXR-GUI benchmark evaluation, synthetic XR GUI data generation, and supervised fine-tuning code.

The repository is organized as a cleaned submission: source code and compact reports are kept under `code/`, while experiment logs, summaries, and evaluation artifacts are kept under `log/`. Large external assets such as model weights, datasets, VM images, Docker images, and most raw screenshots are expected to be prepared locally.

## Repository Structure

```text
GenAI-Group3/
├── code/
│   ├── Grounding/                 BAMI grounding and evaluation utilities
│   ├── Agent/                     OSWorld agent baselines and BAMI variants
│   └── XR/                        EgoXR-GUI evaluation, data synthesis, and SFT
└── log/
    ├── Grounding/                 Grounding log placeholder and output convention
    ├── Agent/                     OSWorld baseline and BAMI-agent reports/results
    └── XR/                        EgoXR-GUI evaluation JSONs and SFT artifacts
```

## Components

### 1. Grounding

Code: [`code/Grounding/`](code/Grounding/)  
Logs: [`log/Grounding/`](log/Grounding/)

The Grounding part contains BAMI grounding code for GUI grounding and ScreenSpot-Pro evaluation. It includes:

- `FeedCoG/`: evaluation pipeline and judging utilities.
- `group3GUI/scripts/`: environment checks and run scripts.
- `group3GUI/requirements-holo-bami.txt`: Python dependencies for the Holo/BAMI setup.

Basic usage:

```bash
cd code/Grounding
pip install -r group3GUI/requirements-holo-bami.txt
bash group3GUI/scripts/check_env.sh
bash group3GUI/scripts/run_holo2_bami.sh 8B
```

Required models and datasets should be configured locally. Common environment variables include `GROUNDING_MODEL`, `LOCAL_JUDGE_MODEL`, `LOCAL_JUDGE_BASE_MODEL`, and `SCREENSPOT_ROOT`.

### 2. Agent

Code: [`code/Agent/`](code/Agent/)  
Logs and reports: [`log/Agent/`](log/Agent/)

The Agent part evaluates GUI agents on OSWorld and adds BAMI-based action refinement.

Main code entries:

- [`code/Agent/baseline/`](code/Agent/baseline/): UI-TARS and AgentS3 OSWorld baseline code.
- [`code/Agent/uitars15_v2_bami.py`](code/Agent/uitars15_v2_bami.py): `UITarsBamiAgent`, which refines UI-TARS coordinate predictions through BAMI-style crop, re-ground, and judge steps.
- [`code/Agent/run_multienv_uitars_bami.py`](code/Agent/run_multienv_uitars_bami.py): multi-environment OSWorld runner for UI-TARS + BAMI.
- [`code/Agent/AgentS3+BAMI/`](code/Agent/AgentS3+BAMI/): AgentS3-BAMI implementation and CLI entry.

OSWorld results summarized in the included README/report files:

| System | Setting | Test Set | Result |
|---|---|---:|---:|
| UI-TARS baseline | UI-TARS-1.5-7B, local vLLM | 369 tasks | 10.8% success, 40/369 full success |
| AgentS3 baseline | GPT-4o planner + UI-TARS-1.5-7B grounding | 361 tasks | 19.9% in baseline summary; detailed report records 20.9% score rate |
| UI-TARS + BAMI | UI-TARS-1.5 series + BAMI refinement | 361 tasks | 12.7%, 46/361 total score |

For detailed configuration, failure analysis, and per-application breakdowns, see:

- [`code/Agent/baseline/README.md`](code/Agent/baseline/README.md)
- [`log/Agent/baseline/README.md`](log/Agent/baseline/README.md)
- [`log/Agent/baseline/uitars/REPORT.md`](log/Agent/baseline/uitars/REPORT.md)
- [`log/Agent/baseline/agents3/REPORT.md`](log/Agent/baseline/agents3/REPORT.md)
- [`log/Agent/uitars_bami/README.md`](log/Agent/uitars_bami/README.md)

AgentS3-BAMI requires external services such as a UI-TARS vLLM server, a BAMI grounding server, Docker/OSWorld environments, and LLM API access. See [`code/Agent/AgentS3+BAMI/README.md`](code/Agent/AgentS3+BAMI/README.md) for command examples.

### 3. XR

Code: [`code/XR/`](code/XR/)  
Logs and results: [`log/XR/`](log/XR/)

The XR part is based on EgoXR-GUI, a benchmark for GUI grounding in physical-digital extended reality. It includes:

- [`code/XR/evaluation/`](code/XR/evaluation/): evaluation scripts for Direct, Spatial, and Semantic Grounding.
- [`code/XR/sys_finetune/`](code/XR/sys_finetune/): synthetic data generation, annotation, and supervised fine-tuning pipeline.
- [`code/XR/sys_finetune/VLM_annotation/`](code/XR/sys_finetune/VLM_annotation/): VLM-driven XR GUI data generator.
- `XR_SFT/`: Qwen3-VL style SFT scripts, LoRA merging utilities, and training config.

The EgoXR-GUI dataset is referenced at:

```text
https://huggingface.co/datasets/Anonymous114/EgoXR-GUI/
```

Evaluation entry points include:

```bash
cd code/XR/evaluation
bash run_models.sh
bash run_finetune_test.sh
bash run_task_router.sh
```

Synthetic XR GUI data generation:

```bash
cd code/XR/sys_finetune/VLM_annotation
python -m xr_gui_grounding --input /path/to/image.jpg --output ./outputs
python -m xr_gui_grounding --input /path/to/images --output ./outputs --max-apps 3
```

The XR logs include evaluation JSON files for multiple models under [`log/XR/evaluation/`](log/XR/evaluation/), plus SFT-related artifacts under [`log/XR/sys_finetune/`](log/XR/sys_finetune/).

## External Assets

The repository does not include all heavyweight runtime assets. Prepare these separately as needed:

- OSWorld test configs, VM image, Docker image, and desktop environment assets.
- UI-TARS-1.5-7B, Holo2, Qwen3-VL, local judge models, and LoRA/checkpoint weights.
- EgoXR-GUI dataset files.
- API keys for external VLM/LLM providers where required.
- Runtime output directories for screenshots, videos, and full raw logs.

## Reading Guide

Start from the component README most relevant to your task:

- Grounding setup: [`code/Grounding/README.md`](code/Grounding/README.md)
- Agent baselines: [`code/Agent/baseline/README.md`](code/Agent/baseline/README.md)
- UI-TARS+BAMI runner: [`code/Agent/README_run_multienv_uitars_bami.md`](code/Agent/README_run_multienv_uitars_bami.md)
- UI-TARS+BAMI agent: [`code/Agent/README_uitars15_v2_bami.md`](code/Agent/README_uitars15_v2_bami.md)
- AgentS3+BAMI: [`code/Agent/AgentS3+BAMI/README.md`](code/Agent/AgentS3+BAMI/README.md)
- XR overview: [`code/XR/README.md`](code/XR/README.md)
- XR evaluation: [`code/XR/evaluation/README.md`](code/XR/evaluation/README.md)
- XR data synthesis and SFT: [`code/XR/sys_finetune/README.md`](code/XR/sys_finetune/README.md)
