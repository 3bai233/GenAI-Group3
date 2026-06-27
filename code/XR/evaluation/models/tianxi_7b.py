import re
from typing import Optional, Tuple, Union

import torch
from PIL import Image
from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration


class TianXi7BModel:
    def __init__(self):
        self.model = None
        self.processor = None
        self.max_pixels = 3840 * 2160
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generation_config = {
            "temperature": 0.0,
            "max_new_tokens": 64,
        }

    def set_generation_config(self, **kwargs):
        self.generation_config.update(kwargs)

    def load_model(self, model_name_or_path: str, max_pixels: int = 3840 * 2160):
        self.max_pixels = max_pixels

        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                attn_implementation="flash_attention_2",
                device_map="auto",
            )
        except Exception:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )

        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_name_or_path)

    def _process_image(self, image: Image.Image) -> Tuple[Image.Image, float, Tuple[int, int]]:
        w, h = image.size
        pixels = w * h
        scale = 1.0

        if pixels > self.max_pixels:
            scale = (self.max_pixels / pixels) ** 0.5
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        else:
            nw, nh = w, h

        # pad to multiple of 28
        pw = ((nw + 27) // 28) * 28
        ph = ((nh + 27) // 28) * 28

        resized = image.resize((nw, nh), Image.LANCZOS)
        padded = Image.new("RGB", (pw, ph), (0, 0, 0))
        padded.paste(resized, (0, 0))
        return padded, scale, (pw, ph)

    def _build_text(self, padded_size: Tuple[int, int], instruction: str) -> str:
        x, y = padded_size
        system_prompt = "You are a helpful assistant."
        user_prompt = (
            f"The image is a screenshot of a computer or mobile phone interface, "
            f"with a resolution of {x}x{y}. Please provide the coordinates of the object "
            f"to be operated according to the command, which is as follows: {instruction}.\n"
        )
        user_prompt_repeat = (
            f"\nRepeat the task again for you:\nPlease provide the coordinates of the object "
            f"to be operated according to the command, which is as follows: {instruction}. "
            f"You must output in the following format: <|box_start|>(x1,y1),(x2,y2)<|box_end|>\n"
        )

        message = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image", "image": "placeholder"},
                    {"type": "text", "text": user_prompt_repeat},
                ],
            },
        ]
        return self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    def _extract_point_px(self, output_text: str) -> Optional[Tuple[float, float]]:
        # 1) bbox: (x1,y1),(x2,y2) -> center
        pairs = re.findall(r"\((\d+),\s*(\d+)\)", output_text)
        if len(pairs) >= 2:
            x1, y1 = map(float, pairs[0])
            x2, y2 = map(float, pairs[1])
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0

        # 2) point: [x,y]
        m = re.search(r"\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]", output_text)
        if m:
            return float(m.group(1)), float(m.group(2))

        return None

    def ground_only_positive(self, instruction: str, image: Union[str, Image.Image]):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        ow, oh = image.size
        img_rgb, scale, padded_size = self._process_image(image)
        text = self._build_text(padded_size, instruction)

        inputs = self.processor(
            text=[text],
            images=[img_rgb],
            max_length=40000,
            truncation=False,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        do_sample = self.generation_config.get("temperature", 0.0) > 0
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.generation_config.get("max_new_tokens", 64),
            do_sample=do_sample,
            temperature=self.generation_config.get("temperature", 0.0) if do_sample else None,
        )
        generated_ids = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True
        )[0]

        point_px = self._extract_point_px(output_text)
        if point_px is None:
            return {"point": None, "raw_response": output_text}

        x, y = point_px
        if scale != 1.0:
            x, y = x / scale, y / scale

        point = [max(0.0, min(1.0, x / ow)), max(0.0, min(1.0, y / oh))]
        return {"point": point, "raw_response": output_text}

    def ground_allow_negative(self, instruction: str, image: Union[str, Image.Image]):
        out = self.ground_only_positive(instruction, image)
        if out["point"] is None:
            return {"result": "negative", "point": None, "raw_response": out["raw_response"]}
        return {"result": "positive", "point": out["point"], "raw_response": out["raw_response"]}