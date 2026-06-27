# EgoXR-GUI: Benchmarking GUI Grounding in Physical-Digital Extended Reality

This repository contains the official code for the paper **EgoXR-GUI: Benchmarking GUI Grounding in Physical-Digital Extended Reality**. 
It provides the full evaluation suite for the EgoXR-GUI benchmark as well as the automated data synthesis pipeline and model fine-tuning (SFT) scripts.

## Directory Structure

- [`evaluation/`](./evaluation/): Contains the evaluation scripts for testing varying Multi-modal Large Language Models (MLLMs) and GUI grounding models on the EgoXR-GUI benchmark. It includes the implementations of prevailing foundation models and the Task Router agentic framework.
- [`sys_finetune/`](./sys_finetune/): Contains the automated data synthesis pipeline leveraging VLMs to iteratively generate semantic instructions and BBox coordinates, as well as the supervised fine-tuning (SFT) scripts used to significantly enhance the semantic grounding performance.

Please refer to the `README.md` files within each directory for tailored instructions on reproducing the experimental results or using the fine-tuning pipeline.