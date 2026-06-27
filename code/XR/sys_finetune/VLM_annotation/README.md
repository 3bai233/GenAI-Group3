# XR GUI Grounding Data Generator

Generate training data for XR GUI semantic grounding based on an egocentric image. The pipeline follows four steps:

1. Scene understanding with a VLM (scene, activity, objects).
2. Candidate UI app generation with VLM.
3. XR screenshot prompt + image generation.
4. VLM-based annotation generation (direct + semantic instructions and bboxes).

## Requirements

- Python 3.10+
- DashScope API key for Qwen-VL
- Image generation API key (laozhang.ai)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file (see `config.example.env`):

```bash
DASHSCOPE_API_KEY=sk-...
OPENAI_API_KEY=sk-...
OPENAI_API_URL=...
```

## Usage

Single image:

```bash
python -m xr_gui_grounding --input /path/to/image.jpg --output ./outputs
```

Directory input:

```bash
python -m xr_gui_grounding --input /path/to/images --output ./outputs --max-apps 3
```

Common options:

- `--vlm-model`: VLM model name (default: `qwen-vl-max`).
- `--max-apps`: Number of UI candidates to generate (default: 3).
- `--app-index`: Which candidate to use for image/annotations (default: 0).
- `--skip-image-gen`: Skip image generation.
- `--xr-image`: Use an existing XR image for annotation.
- `--skip-annotations`: Skip annotation generation.
- `--instruction-language`: Instruction language (default: `Chinese`).

## Output Structure

Each input image creates a subfolder under `--output`:

```
outputs/
  IMG_0001_20260504_120000/
    scene.json
    ui_candidates.json
    image_prompt.json
    xr_image.png
    annotations.json
    manifest.json
```

## Notes

- The VLM prompts are designed to return strict JSON for reliable parsing.
- Bounding boxes are expected in pixel coordinates for 2048x1152.
