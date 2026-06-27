import base64
import os
import re
from io import BytesIO
from pathlib import Path

from PIL import Image
from openai import OpenAI

DEFAULT_MODEL = "doubao-seed-1-6-vision-250815"
DEFAULT_BASE_URL = "YOUR_URL"

GROUNDING_PROMPT_POS = """You are a GUI grounding agent.
## Output Format
Action: click(point='<point>x y</point>')

## Rules
- Use pixel coordinates in the original image (width, height).
- Output only one Action line.

## User Instruction
{instruction}
"""

GROUNDING_PROMPT_NEG = """You are a GUI grounding agent.
## Output Format
Action: click(point='<point>x y</point>')
# If the target does not exist, output:
Action: finished(content='negative')

## Rules
- Use pixel coordinates in the original image (width, height).
- Output only one Action line.

## User Instruction
{instruction}
"""

_CLICK_RE = re.compile(
    r"click\s*\(\s*point\s*=\s*['\"]<point>\s*([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s*</point>['\"]\s*\)",
    re.IGNORECASE,
)
_FINISH_RE = re.compile(
    r"finished\s*\(\s*content\s*=\s*['\"](.*?)['\"]\s*\)",
    re.IGNORECASE,
)


class Seed1_5VLModel:
    def __init__(self):
        self.client = None
        self.model_name = DEFAULT_MODEL
        self.base_url = os.getenv("ARK_BASE_URL", DEFAULT_BASE_URL)
        self.api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("ARK_API_KEY")
            or os.getenv("VOLC_API_KEY")
        )
        self.temperature = 0.0
        self.max_new_tokens = 256
        self.top_p = None
        self.stream = os.getenv("SEED1_5VL_STREAM", "false").lower() in ("1", "true", "yes")

    def load_model(self, model_name_or_path=None, base_url=None, api_key=None):
        self.model_name = model_name_or_path or os.getenv("SEED1_5VL_MODEL", self.model_name)
        self.base_url = base_url or self.base_url
        self.api_key = api_key or self.api_key
        if not self.api_key:
            raise ValueError("Please set OPENAI_API_KEY/ARK_API_KEY/VOLC_API_KEY for Seed1.5-VL.")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def set_generation_config(self, temperature=0, max_new_tokens=256, **kwargs):
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = kwargs.get("top_p", self.top_p)

    def ground_only_positive(self, instruction, image):
        response_text, img_size = self._infer(instruction, image, allow_negative=False)
        point = self._parse_click_point(response_text, img_size)
        return {
            "point": point,
            "bbox": None,
            "result": "positive" if point else "wrong_format",
            "raw_response": response_text,
        }

    def ground_allow_negative(self, instruction, image):
        response_text, img_size = self._infer(instruction, image, allow_negative=True)
        point = self._parse_click_point(response_text, img_size)
        if point:
            result = "positive"
        elif self._is_negative(response_text):
            result = "negative"
        else:
            result = "wrong_format"
        return {
            "point": point,
            "bbox": None,
            "result": result,
            "raw_response": response_text,
        }

    def _infer(self, instruction, image_path, allow_negative):
        if self.client is None:
            self.load_model()

        prompt = self._build_prompt(instruction or "", allow_negative)
        b64, img_format, width, height = self._encode_image(image_path)

        messages = [
            {"role": "user", "content": prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{img_format};base64,{b64}"},
                    }
                ],
            },
        ]

        response_text = self._chat(messages)
        return response_text, (width, height)

    def _chat(self, messages):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "stream": self.stream,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p

        if self.stream:
            response_text = ""
            for chunk in self.client.chat.completions.create(**payload):
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    response_text += delta.content
            return response_text
        else:
            resp = self.client.chat.completions.create(**payload)
            return (resp.choices[0].message.content or "")

    def _build_prompt(self, instruction, allow_negative):
        if allow_negative:
            return GROUNDING_PROMPT_NEG.format(instruction=instruction)
        return GROUNDING_PROMPT_POS.format(instruction=instruction)

    def _encode_image(self, image_path):
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        suffix = image_path.suffix.lower().lstrip(".")
        with Image.open(image_path) as img:
            width, height = img.size
            if suffix in {"jpg", "jpeg", "png", "webp"}:
                with open(image_path, "rb") as f:
                    data = f.read()
                img_format = "jpeg" if suffix == "jpg" else suffix
            else:
                buf = BytesIO()
                img.convert("RGB").save(buf, format="PNG")
                data = buf.getvalue()
                img_format = "png"

        b64 = base64.b64encode(data).decode("utf-8")
        return b64, img_format, width, height

    def _parse_click_point(self, text, img_size):
        match = _CLICK_RE.search(text or "")
        if not match or not img_size:
            return None
        x = float(match.group(1))
        y = float(match.group(2))
        return self._normalize_point(x, y, img_size[0], img_size[1])

    def _normalize_point(self, x, y, width, height):
        if width <= 0 or height <= 0:
            return None

        # Heuristic: support normalized (0~1) or 0~1000 coordinate outputs
        if 0 <= x <= 1 and 0 <= y <= 1:
            nx, ny = x, y
        elif 0 <= x <= 1000 and 0 <= y <= 1000 and max(width, height) > 1200:
            nx, ny = x / 1000.0, y / 1000.0
        else:
            nx, ny = x / width, y / height

        nx = min(max(nx, 0.0), 1.0)
        ny = min(max(ny, 0.0), 1.0)
        return [nx, ny]

    def _is_negative(self, text):
        m = _FINISH_RE.search(text or "")
        if m:
            content = (m.group(1) or "").strip().lower()
            if any(k in content for k in ["negative", "no target", "none", "不存在", "没有目标", "无目标"]):
                return True
        if re.search(r"finished\s*\(\s*content\s*=\s*['\"].*(negative|no target|none|不存在|没有目标|无目标)", text or "", re.I):
            return True
        return False