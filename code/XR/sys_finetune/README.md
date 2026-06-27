# EgoXR-GUI Data Synthesis & Fine-Tuning 

This directory contains the automated data synthesis pipeline and the supervised fine-tuning (SFT) implementation. These modules collectively address the explicit scarcity of Semantic Grounding training data within XR contexts by procedurally generating structured semantic data pairs and using them to adapt foundations models.

## Module Workflow

### 1. `bbox_annotation/`
Tools and execution scripts designed for recording UI components explicitly with spatial parameters. 
- Performs dense localization and generates structured bounding box configurations representing elements natively rendered in target XR environments.

### 2. `VLM_annotation/`
The natural language synthesis and instruction formatting pipeline. 
- Automatically engages large vision-language API models to generate context-grounded semantic descriptions, instructions, and reasoning chains matching localized bounding elements.


### 3. `XR_SFT/`
Post-processing Supervised Fine-Tuning logic primarily configured for PyTorch and accelerating architectures.
- `train.py` & `run_train_ddp.sh`: Conduct Distributed Data Parallel training on models (e.g. Qwen3-VL-8B).
- `merge_lora.py`: A utility dedicated to merging previously trained LoRA adapter weights natively into the base VLM. 
- `config/train_config.json`: Master hyperparameter specifications dictating SFT execution routines and optimization.