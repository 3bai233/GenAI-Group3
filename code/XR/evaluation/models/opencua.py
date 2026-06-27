import json
import os
import re
from typing import Any

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


SYSTEM_PROMPT = (
    "You are a GUI agent. You are given a task and a screenshot of the screen. "
    "You need to perform a series of pyautogui actions to complete the task."
)

MAX_PIXELS = 3840 * 1920


def _resize_image_if_needed(image: Image.Image, max_pixels: int = MAX_PIXELS) -> Image.Image:
    W, H = image.width, image.height
    if W * H <= max_pixels:
        return image
    scale = (max_pixels / (W * H)) ** 0.5
    new_W = int(W * scale)
    new_H = int(H * scale)
    return image.resize((new_W, new_H), Image.LANCZOS)


def _parse_point_from_response(text: str, img_size: list[int] | None) -> list[float] | None:
    if img_size is None:
        return None
    W, H = img_size[0], img_size[1]

    m = re.search(r'pyautogui\.click\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)', text)
    if m:
        return [float(m.group(1)) / W, float(m.group(2)) / H]

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            coord = obj.get("coordinate") or obj.get("point")
            if coord and len(coord) == 2:
                return [float(coord[0]) / W, float(coord[1]) / H]
            if "x" in obj and "y" in obj:
                return [float(obj["x"]) / W, float(obj["y"]) / H]
    except (json.JSONDecodeError, TypeError):
        pass

    m = re.search(r'[\(\[]\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*[\)\]]', text)
    if m:
        return [float(m.group(1)) / W, float(m.group(2)) / H]

    nums = re.findall(r'-?[\d]+(?:\.[\d]+)?', text)
    if len(nums) >= 2:
        return [float(nums[0]) / W, float(nums[1]) / H]

    return None


MEDIA_BEGIN_ID   = 151661  # <|media_begin|>
MEDIA_CONTENT_ID = 151662  # <|media_content|>
MEDIA_END_ID     = 151663  # <|media_end|>

IM_SYSTEM_ID    = 151653  # <|im_system|>
IM_USER_ID      = 151646  # <|im_user|>
IM_ASSISTANT_ID = 151647  # <|im_assistant|>
IM_END_ID       = 151645  # <|im_end|>


class OpenCUAModel:
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None

    def load_model(self, model_name_or_path: str = "OpenCUA/OpenCUA-7B"):
        print(f"Loading OpenCUA model from {model_name_or_path} ...")

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=False,
        )
        self.tokenizer = self.processor.tokenizer

        try:
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        except TypeError:
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        print(f"OpenCUA model loaded from {model_name_or_path}")

    def _run_inference(self, instruction: str, image: Image.Image) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "<image>"},
                    {"type": "text", "text": instruction},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )

        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                if v.dtype.is_floating_point:
                    inputs[k] = v.to(self.model.device, dtype=torch.bfloat16)
                else:
                    inputs[k] = v.to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        new_ids = generated_ids[:, prompt_len:]
        output_text = self.processor.batch_decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip()

    def _infer(self, instruction: str, pil_image: Image.Image) -> dict[str, Any]:
        scaled_size = [pil_image.width, pil_image.height]
        raw_response = self._run_inference(instruction, pil_image)
        point = _parse_point_from_response(raw_response, scaled_size)
        return {
            "result": "positive",
            "format": "point",
            "raw_response": raw_response,
            "bbox": None,
            "point": point,
        }

    def ground_only_positive(self, instruction: str, image) -> dict[str, Any]:
        if isinstance(image, str):
            assert os.path.exists(image) and os.path.isfile(image), \
                f"Invalid input image path: {image}"
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise ValueError("image must be a file path or PIL Image.")

        scaled_image = _resize_image_if_needed(pil_image, MAX_PIXELS)
        return self._infer(instruction, scaled_image)

    def ground_allow_negative(self, instruction: str, image) -> dict[str, Any]:
        response = self.ground_only_positive(instruction, image)
        if response["point"] is None:
            response["result"] = "negative"
        return response