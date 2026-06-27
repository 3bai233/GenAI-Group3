from __future__ import annotations

import base64
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PIL import Image


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_image_as_data_url(image_path: str | Path) -> str:
    image_path = Path(image_path)
    mime = _guess_mime(image_path)
    with image_path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _guess_mime(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return "image/png"


def extract_json_block(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if text.startswith("{") or text.startswith("["):
        return text
    match = re.search(r"[{\[]", text)
    if not match:
        return text
    start = match.start()
    for end in range(len(text), start, -1):
        snippet = text[start:end].strip()
        if not snippet:
            continue
        try:
            json.loads(snippet)
            return snippet
        except json.JSONDecodeError:
            continue
    return text


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def normalize_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    x2 = max(1, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(1, min(y2, height))
    if x1 >= x2:
        x1 = max(0, x2 - 1)
    if y1 >= y2:
        y1 = max(0, y2 - 1)
    return [x1, y1, x2, y2]


def get_image_size(image_path: str | Path) -> tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}


def load_env_file(env_path: Optional[str]) -> None:
    if not env_path:
        return
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"env file not found: {env_path}")
    for line in Path(env_path).read_text(encoding="utf-8").splitlines():
        if not line or line.strip().startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())
