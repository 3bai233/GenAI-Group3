import os
from typing import Any, Literal

import torch
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize


class ClickCoordinates(BaseModel):
    x: int = Field(ge=0, le=1000, description="The x coordinate, normalized between 0 and 1000.")
    y: int = Field(ge=0, le=1000, description="The y coordinate, normalized between 0 and 1000.")


class Holo2Model:
    def __init__(self):
        self.model = None
        self.processor = None
        self.prompt_template = (
            "Localize an element on the GUI image according to the provided target and output a click position.\n"
            " * You must output a valid JSON following the format: {schema}\n"
            " Your target is:"
        )

    def load_model(self, model_name_or_path: str = "Hcompany/Holo2-8B"):
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        print(f"Holo2 model loaded from {model_name_or_path}")

    def set_generation_config(self, **kwargs):
        pass

    def _build_messages(self, instruction: str, image: Image.Image) -> list[dict[str, Any]]:
        prompt = self.prompt_template.format(schema=ClickCoordinates.model_json_schema())
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"{prompt}\n{instruction}"},
                ],
            }
        ]

    def _resize_image(self, image: Image.Image) -> Image.Image:
        image_processor_config = self.processor.image_processor
        resized_height, resized_width = smart_resize(
            image.height,
            image.width,
            factor=image_processor_config.patch_size * image_processor_config.merge_size,
            min_pixels=image_processor_config.size.get("shortest_edge", None),
            max_pixels=image_processor_config.size.get("longest_edge", None),
        )
        return image.resize((resized_width, resized_height), resample=Image.Resampling.LANCZOS)

    def _parse_reasoning(self, generated_ids: torch.Tensor) -> tuple[str, str]:
        all_ids = generated_ids[0].tolist()
        try:
            think_start_index = all_ids.index(151667)
        except ValueError:
            # 没有 thinking token，直接解码全部
            content = self.processor.decode(all_ids, skip_special_tokens=True).strip("\n")
            return content, ""

        try:
            think_end_index = all_ids.index(151668)
        except ValueError:
            think_end_index = len(all_ids)

        thinking_content = self.processor.decode(
            all_ids[think_start_index + 1 : think_end_index],
            skip_special_tokens=True,
        ).strip("\n")
        content = self.processor.decode(
            all_ids[think_end_index + 1 :],
            skip_special_tokens=True,
        ).strip("\n")
        return content, thinking_content

    def _run_inference(self, messages: list[dict], processed_image: Image.Image) -> str:
        # Note: thinking=False for localization
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            thinking=False,
        )
        inputs = self.processor(
            text=[text_prompt],
            images=[processed_image],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=32)
        content, _ = self._parse_reasoning(generated_ids)
        return content

    def ground_only_positive(self, instruction: str, image) -> dict[str, Any]:
        if isinstance(image, str):
            assert os.path.exists(image) and os.path.isfile(image), "Invalid input image path."
            image = Image.open(image).convert("RGB")
        assert isinstance(image, Image.Image), "Invalid input image."

        processed_image = self._resize_image(image)
        messages = self._build_messages(instruction, processed_image)
        response = self._run_inference(messages, processed_image)

        try:
            click_action = ClickCoordinates.model_validate_json(response)
        except ValidationError:
            return {
                "result": "positive",
                "format": "x1y1x2y2",
                "raw_response": response,
                "bbox": None,
                "point": None,
            }

        relative_x = click_action.x / 1000.0
        relative_y = click_action.y / 1000.0

        return {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": [relative_x, relative_y],
        }

    def ground_allow_negative(self, instruction: str, image) -> dict[str, Any]:
        return self.ground_only_positive(instruction, image)