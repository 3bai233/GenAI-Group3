import os
import re
import torch
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
# Ensure UIVenusGroundV15 class can be imported
from ui_venus1_5_gd import UIVenusGroundV15

class UIVenusBBoxVisualizer(UIVenusGroundV15):
    """Extend original model class with bbox output and visualization support"""

    def inference_bbox(self, instruction, image_path, visualize=False, save_path="output_vis.png"):
        """
        Generate bounding box and optionally visualize it.

        Args:
            instruction: Instruction for UI element
            image_path: Image path
            visualize: Whether to draw and save result on image
            save_path: Path to save visualized image

        Returns:
            dict: Contains raw_response and bbox (coordinates normalized to 0-1000 relative to original image size)
        """
        if isinstance(image_path, str):
            assert os.path.exists(image_path), f"Invalid image path: {image_path}"
            image = Image.open(image_path).convert("RGB")
        elif isinstance(image_path, Image.Image):
            image = image_path.copy()
            # If it's an object, for passing to processor it's best to convert to temp path
            # or pass object directly to process_vision_info
            # Qwen-vl message format can directly accept PIL Image
        else:
            raise ValueError("image must be a file path (str) or PIL.Image.Image object")

        # 1. Adjust prompt, require model to output [x1, y1, x2, y2]
        if instruction.endswith('.'):
            instruction = instruction[:-1]

        prompt = (
            "Output the bounding box of the position corresponding to the following instruction: \n{}. \n\n"
            "The output should just be the coordinates of a bounding box, in the format [x1,y1,x2,y2]. "
            "Additionally, if the task is infeasible (e.g., the task is not related to the image), the output should be [-1,-1,-1,-1]."
        )
        full_prompt = prompt.format(instruction)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": full_prompt},
                ],
            }
        ]

        # 2. Process and run inference
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        raw_output = output_text[0].strip()

        # 3. Parse model output bbox [x1, y1, x2, y2]
        pattern_bbox = r"\[\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*,\s*-?\d+\s*\]"
        bbox = None
        if re.fullmatch(pattern_bbox, raw_output, re.DOTALL):
            bbox = eval(raw_output)

        # 4. Visualization
        if visualize and bbox is not None and bbox != [-1, -1, -1, -1]:
            # Model usually outputs absolute coordinates mapped to 1000x1000 scale
            width, height = image.size
            x1 = int((bbox[0] / 1000.0) * width)
            y1 = int((bbox[1] / 1000.0) * height)
            x2 = int((bbox[2] / 1000.0) * width)
            y2 = int((bbox[3] / 1000.0) * height)

            draw = ImageDraw.Draw(image)
            # Draw a red rectangle with width 3
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            image.save(save_path)
            print(f"Visualization saved to: {save_path}")

        return {
            "result": "infeasible" if bbox == [-1, -1, -1, -1] else "positive" if bbox else "wrong_format",
            "raw_response": raw_output,
            "bbox": bbox  # Returns [x1, y1, x2, y2] (based on 1000 percentage system)
        }

# ============================
# Test case (supports command line arguments and multi-JSON batch processing)
# ============================
if __name__ == "__main__":
    import json
    import argparse
    import shutil
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Batch generate bounding boxes for UI elements from multiple JSON files.")
    parser.add_argument("--model_path", type=str, default="YOUR_MODEL_PATH", help="Path to the model.")
    parser.add_argument("--json_files", type=str, nargs="+", required=True, help="List of JSON files to process.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., cuda, cpu).")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization of bounding boxes.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save output JSON and images. Default is alongside the input JSON.")
    args = parser.parse_args()

    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Initialize model
    visualizer = UIVenusBBoxVisualizer()
    visualizer.load_model(model_name_or_path=args.model_path, device=args.device)

    # Iterate through all input JSON files
    for json_path in args.json_files:
        base_dir = os.path.dirname(json_path) or "."
        out_dir = args.output_dir if args.output_dir else base_dir

        # If output folder specified, create subfolders
        if args.output_dir:
            orig_images_dir = os.path.join(out_dir, "orig_images")
            vis_images_dir = os.path.join(out_dir, "vis_images")
            os.makedirs(orig_images_dir, exist_ok=True)
            os.makedirs(vis_images_dir, exist_ok=True)
        else:
            orig_images_dir = base_dir
            vis_images_dir = base_dir

        print(f"\n{'='*50}\nProcessing JSON file: {json_path}\n{'='*50}")

        if not os.path.exists(json_path):
            print(f"JSON file not found: {json_path}, skipping...")
            continue

        file_name = os.path.basename(json_path)
        name_only, file_ext = os.path.splitext(file_name)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item_idx, item in tqdm(enumerate(data), total=len(data), desc=f"Processing {name_only}"):
            rel_image_path = item.get("xr_image_path", "")
            image_path = os.path.join(base_dir, rel_image_path)

            if not os.path.exists(image_path):
                tqdm.write(f"Image not found: {image_path}")
                continue

            img_basename = os.path.basename(image_path)
            # Prevent name collision, add JSON filename and item_idx as unique prefix
            unique_img_name = f"{name_only}_item_{item_idx}_{img_basename}"

            if args.output_dir:
                dest_orig_path = os.path.join(orig_images_dir, unique_img_name)
                # Copy original image to orig_images folder
                shutil.copy2(image_path, dest_orig_path)
                # Write relative path in json
                item["output_orig_image_path"] = os.path.join("orig_images", unique_img_name)

            for ann_idx, ann in enumerate(item.get("annotations", [])):
                instruction = ann.get("direct_instruction_en")
                if not instruction:
                    continue

                # To avoid overwriting, add JSON name and current annotation index to filename
                if args.output_dir:
                    save_path = os.path.join(vis_images_dir, f"{name_only}_item_{item_idx}_vis_ann{ann_idx}.png")
                else:
                    save_path = f"{os.path.splitext(image_path)[0]}_{name_only}_vis_ann{ann_idx}.png"

                res = visualizer.inference_bbox(
                    instruction=instruction,
                    image_path=image_path,
                    visualize=args.visualize,
                    save_path=save_path
                )
                # Save results back to dict for review and calculation
                ann["predicted_bbox"] = res.get("bbox")

        # Save results with predicted bboxes as new json
        output_json = os.path.join(out_dir, f"{name_only}_with_bbox{file_ext}")

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[OK] JSON processing complete, updated info saved to: {output_json}\n")
