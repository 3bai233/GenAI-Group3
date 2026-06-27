import torch
from transformers import AutoProcessor
import re
import os
from PIL import Image
from tqdm import tqdm
from qwen_vl_utils import smart_resize
import multiprocessing as mp

mp.set_start_method('spawn', force=True)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def parse_coordinates(raw_string):
    m = re.search(r'"coordinate"\s*:\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]', raw_string)
    if m:
        return float(m.group(1)), float(m.group(2))

    matches = re.findall(r'\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]', raw_string)
    if not matches:
        return None
    x, y = matches[0]
    return float(x), float(y)


def get_qwen3_vl_prompt_msg(image, instruction):
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": """You are a GUI grounding agent. 
## Task
Given a screenshot and the user's grounding instruction. Your task is to accurately locate a UI element based on the user's instructions.
First, you should carefully examine the screenshot and analyze the user's instructions, translate the user's instruction into an effective reasoning process, and then provide the final coordinate.
## Output Format
Return a json object with a reasoning process in <grounding_think></grounding_think> tags, a [x,y] format coordinate within <answer></answer> XML tags:
<grounding_think>...</grounding_think>
<answer>
{"coordinate": [x,y]}
</answer>
## Input instruction
"""
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction + "\n"},
                {"type": "image", "image": image},
            ]
        }
    ]
    return messages


from vllm import SamplingParams


class CustomQwen3_VL_VLLM_Model():
    def __init__(self):
        from multiprocessing import current_process
        process = current_process()
        if process.daemon:
            print("vLLM cannot be started in daemon process.")

        self.generation_config = {
            "temperature": 0.0,
            "max_tokens": 256,
        }
        self.coord_mode = "auto"  # auto | grid1000 | pixel | normalized

    def load_model(self, model_name_or_path="Qwen/Qwen3-VL-30B-A3B-Instruct", max_pixels=3840 * 2160):
        from vllm import LLM
        self.max_pixels = max_pixels

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            do_resize=False,
        )

        tp_size = max(1, torch.cuda.device_count())
        self.model = LLM(
            model=model_name_or_path,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=0.90,
            max_model_len=32768,
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": max_pixels,
            },
        )

    def set_generation_config(self, **kwargs):
        self.generation_config.update(kwargs)

    def _normalize_point(self, x, y, w, h):
        mode = self.coord_mode
        if mode == "normalized":
            return [x, y]
        if mode == "grid1000":
            return [x / 1000.0, y / 1000.0]
        if mode == "pixel":
            return [x / w, y / h]

        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            return [x, y]
        if 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0:
            return [x / 1000.0, y / 1000.0]
        return [x / w, y / h]

    def ground_only_positive(self, instruction, image, use_guide_text=False):
        if isinstance(image, str):
            assert os.path.exists(image), f"Invalid image path: {image}"
            image = Image.open(image).convert("RGB")
        assert isinstance(image, Image.Image), "Invalid input image."

        resized_h, resized_w = smart_resize(
            image.height,
            image.width,
            factor=14 * 2,
            min_pixels=28 * 28,
            max_pixels=self.max_pixels,
        )
        resized_image = image.resize((resized_w, resized_h))

        messages = get_qwen3_vl_prompt_msg(resized_image, instruction)
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        if use_guide_text:
            prompt += '<tool_call>\n{"name":"grounding","arguments":{"action":"click","coordinate":['

        inputs = [{
            "prompt": prompt,
            "multi_modal_data": {"image": resized_image},
        }]

        generated = self.model.generate(
            inputs,
            sampling_params=SamplingParams(
                temperature=self.generation_config.get("temperature", 0.0),
                max_tokens=self.generation_config.get("max_tokens", 256),
            ),
        )

        raw_text = generated[0].outputs[0].text
        coord = parse_coordinates(raw_text)
        if coord is None:
            return {"point": None, "raw_response": raw_text}

        x, y = coord
        point = self._normalize_point(x, y, resized_w, resized_h)
        return {"point": point, "raw_response": raw_text}

    def batch_ground_only_positive(self, instructions, images, use_guide_text=False):
        assert len(instructions) == len(images), "instructions/images size mismatch"

        batch_inputs = []
        resized_sizes = []

        for instruction, image in tqdm(zip(instructions, images), total=len(instructions)):
            if isinstance(image, str):
                assert os.path.exists(image), f"Invalid image path: {image}"
                image = Image.open(image).convert("RGB")
            assert isinstance(image, Image.Image), "Invalid input image."

            resized_h, resized_w = smart_resize(
                image.height,
                image.width,
                factor=14 * 2,
                min_pixels=28 * 28,
                max_pixels=self.max_pixels,
            )
            resized_image = image.resize((resized_w, resized_h))
            resized_sizes.append((resized_w, resized_h))

            messages = get_qwen3_vl_prompt_msg(resized_image, instruction)
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if use_guide_text:
                prompt += '<tool_call>\n{"name":"grounding","arguments":{"action":"click","coordinate":['

            batch_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": resized_image},
            })

        outputs = self.model.generate(
            batch_inputs,
            sampling_params=SamplingParams(
                temperature=self.generation_config.get("temperature", 0.0),
                max_tokens=self.generation_config.get("max_tokens", 256),
            ),
            use_tqdm=True,
        )

        results = []
        for output, (w, h) in zip(outputs, resized_sizes):
            raw_text = output.outputs[0].text
            coord = parse_coordinates(raw_text)
            point = None
            if coord is not None:
                x, y = coord
                point = self._normalize_point(x, y, w, h)

            results.append({
                "result": "positive",
                "format": "point",
                "raw_response": raw_text,
                "bbox": None,
                "point": point,
            })

        return results