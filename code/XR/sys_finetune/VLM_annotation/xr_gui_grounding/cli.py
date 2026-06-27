from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from xr_gui_grounding.models import PipelineConfig
from xr_gui_grounding.pipeline import collect_images, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XR GUI Grounding Data Generator")
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--vlm-model", default="qwen-vl-max", help="VLM model name")
    parser.add_argument("--max-apps", type=int, default=3, help="Max UI apps to propose")
    parser.add_argument("--num-annotations", type=int, default=5, help="Number of annotations to generate")
    parser.add_argument("--bilingual", action="store_true", help="Generate both Chinese and English instructions")
    parser.add_argument("--max-images", type=int, default=0, help="Max number of images to process in input directory (0 for all)")
    parser.add_argument("--app-index", type=int, default=0, help="Chosen app index")
    parser.add_argument("--image-model", default="gpt-image-2-vip", help="Image model name")
    parser.add_argument("--image-size", default="2048x1152", help="Output image size")
    parser.add_argument("--skip-image-gen", action="store_true", help="Skip image generation")
    parser.add_argument("--xr-image", help="Use existing XR image for annotations")
    parser.add_argument("--skip-annotations", action="store_true", help="Skip annotation generation")
    parser.add_argument(
        "--instruction-language",
        default="Chinese",
        help="Language for instructions in annotations",
    )
    parser.add_argument("--enable-thinking", action="store_true", help="Enable Qwen thinking mode")
    parser.add_argument("--visualize-annotations", action="store_true", help="Draw bboxes on image")
    parser.add_argument("--env", help="Path to .env file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.env:
        load_dotenv(args.env)
    else:
        load_dotenv()
    config = PipelineConfig(
        output_dir=args.output,
        vlm_model=args.vlm_model,
        max_apps=args.max_apps,
        app_index=args.app_index,
        image_model=args.image_model,
        image_size=args.image_size,
        skip_image_gen=args.skip_image_gen,
        skip_annotations=args.skip_annotations,
        xr_image=args.xr_image,
        instruction_language=args.instruction_language,
        enable_thinking=args.enable_thinking,
        visualize_annotations=args.visualize_annotations,
        num_annotations=args.num_annotations,
        bilingual_annotations=args.bilingual,
        max_images=args.max_images,
    )
    images = collect_images(Path(args.input))
    run_pipeline(config, list(images))


if __name__ == "__main__":
    main()
