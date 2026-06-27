from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
import json

from PIL import Image, ImageDraw
from tqdm import tqdm

from xr_gui_grounding.image_client import ImageClient
from xr_gui_grounding.models import PipelineConfig, SceneUnderstanding
from xr_gui_grounding.utils import (
    ensure_dir,
    is_image_file,
    timestamp,
    write_json,
)
from xr_gui_grounding.vlm_client import VlmClient


def collect_images(input_path: str | Path) -> List[Path]:
    input_path = Path(input_path)
    if input_path.is_dir():
        return sorted(p for p in input_path.iterdir() if is_image_file(p))
    if input_path.is_file() and is_image_file(input_path):
        return [input_path]
    raise FileNotFoundError(f"No images found at {input_path}")


def run_pipeline(config: PipelineConfig, images: Iterable[Path]) -> None:
    images_list = list(images)
    if config.max_images > 0:
        images_list = images_list[:config.max_images]
    print(f"Starting pipeline. Tracking {len(images_list)} images")
    output_root = ensure_dir(config.output_dir)
    vlm = VlmClient(api_key=None, model=config.vlm_model, enable_thinking=config.enable_thinking)
    image_client = None if config.skip_image_gen else ImageClient(api_key=None, api_url=None)

    summary_data = []
    processed_images = set()

    for manifest_path in output_root.glob("*/manifest.json"):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            src_img = data.get("source_image")
            if not src_img:
                continue
                
            src_name = Path(src_img).name
            processed_images.add(src_name)
            
            xr_img = data.get("xr_image")
            annotations_data = data.get("annotations")
            
            if annotations_data is not None and xr_img:
                try:
                    rel_xr = str(Path(xr_img).relative_to(output_root))
                except ValueError:
                    rel_xr = Path(xr_img).name
                
                summary_data.append({
                    "source_image": src_name,
                    "xr_image_path": rel_xr,
                    "annotations": annotations_data.get("annotations", [])
                })
        except Exception:
            continue

    images_to_process = [img for img in images_list if img.name not in processed_images]
    skipped_count = len(images_list) - len(images_to_process)
    if skipped_count > 0:
        print(f"Skipping {skipped_count} already processed images. Remaining: {len(images_to_process)}")

    for image_path in tqdm(images_to_process, desc="Processing Images"):
        tqdm.write(f"\n[{timestamp()}] Processing image: {image_path.name}")
        run_id = f"{image_path.stem}_{timestamp()}"
        run_dir = ensure_dir(output_root / run_id)

        tqdm.write("  - [1/4] Analyzing scene...")
        scene = vlm.analyze_scene(str(image_path))
        write_json(run_dir / "scene.json", scene.model_dump())

        tqdm.write("  - [2/4] Generating UI candidates...")
        candidates = vlm.suggest_ui_apps(str(image_path), scene, config.max_apps)
        write_json(run_dir / "ui_candidates.json", candidates.model_dump())

        if not candidates.apps:
            raise RuntimeError("No UI candidates returned by VLM")
        app_index = min(max(config.app_index, 0), len(candidates.apps) - 1)
        chosen = candidates.apps[app_index]
        tqdm.write(f"    Selected UI App: {chosen.app_name}")

        tqdm.write("  - [3/4] Building image prompt & generating XR screenshot...")
        image_prompt = vlm.build_image_prompt(
            str(image_path),
            scene,
            chosen.app_name,
            chosen.ui_elements,
            chosen.required_items,
            chosen.window_position.model_dump(),
        )
        write_json(run_dir / "image_prompt.json", image_prompt.model_dump())

        xr_image_path = None
        if config.xr_image:
            tqdm.write(f"    Using existing XR image: {config.xr_image}")
            xr_image_path = Path(config.xr_image)
        elif not config.skip_image_gen:
            tqdm.write("    Requesting image generation API...")
            image_bytes = image_client.generate_image(
                image_prompt.prompt,
                model=config.image_model,
                size=config.image_size,
                image_path=str(image_path),
            )
            xr_image_path = run_dir / "xr_image.png"
            xr_image_path.write_bytes(image_bytes)
            tqdm.write(f"    Saved generated XR image to: {xr_image_path.name}")
        else:
            tqdm.write("    Skipped image generation.")

        tqdm.write("  - [4/4] Generating annotations...")
        annotations_data = None
        if not config.skip_annotations and xr_image_path:
            annotations = vlm.generate_annotations(
                str(xr_image_path),
                scene,
                chosen.app_name,
                config.instruction_language,
                config.num_annotations,
                config.bilingual_annotations,
            )
            normalized = []
            for item in annotations.annotations:
                normalized.append(item.model_dump())
            annotations_data = {"annotations": normalized}
            write_json(run_dir / "annotations.json", annotations_data)
            tqdm.write(f"    Generated {len(normalized)} annotations.")
            
            # Add to summary data
            summary_data.append({
                "source_image": str(image_path.name),
                "xr_image_path": str(xr_image_path.relative_to(output_root)),
                "annotations": normalized
            })
            
            if config.visualize_annotations:
                tqdm.write("    Skipping visualization as bbox generation is currently disabled.")
        else:
            tqdm.write("    Skipped annotations.")

        _write_manifest(run_dir, image_path, scene, chosen.app_name, xr_image_path, annotations_data)
        tqdm.write(f"    Finished processing. Outputs saved in: {run_dir}")
        
    if summary_data:
        summary_path = output_root / "summary.json"
        write_json(summary_path, summary_data)
        print(f"\nPipeline execution completed. Summary saved to: {summary_path}")
    else:
        print("\nPipeline execution completed.")


def _write_manifest(
    run_dir: Path,
    source_image: Path,
    scene: SceneUnderstanding,
    app_name: str,
    xr_image_path: Path | None,
    annotations_data: dict | None,
) -> None:
    manifest = {
        "source_image": str(source_image),
        "scene": scene.model_dump(),
        "app_name": app_name,
        "xr_image": str(xr_image_path) if xr_image_path else None,
        "annotations": annotations_data,
    }
    write_json(run_dir / "manifest.json", manifest)
