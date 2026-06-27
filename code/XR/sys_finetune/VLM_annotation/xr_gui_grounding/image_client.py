from __future__ import annotations

import base64
import os
from typing import Optional

import requests


class ImageClient:
    def __init__(self, api_key: Optional[str], api_url: Optional[str]) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required")
        self.api_url = api_url or os.getenv("IMAGE_API_URL") or "https://api.openai.com/v1/images/generations"

    def generate_image(self, prompt: str, model: str, size: str, image_path: Optional[str] = None) -> bytes:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        if image_path:
            # Image edit endpoint
            api_url = "https://api.openai.com/v1/images/edits"
            data = {"model": model, "prompt": prompt, "size": size}
            opened_files = []
            files = []
            import mimetypes
            image_mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
            image_file = open(image_path, "rb")
            opened_files.append(image_file)
            files.append(("image", (os.path.basename(image_path), image_file, image_mime)))

            try:
                response = requests.post(api_url, headers=headers, data=data, files=files, timeout=180)
            finally:
                for f in opened_files:
                    f.close()
        else:
            # Standard generation endpoint
            headers["Content-Type"] = "application/json"
            payload = {"model": model, "prompt": prompt, "size": size}
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=180)

        if response.status_code != 200:
            raise RuntimeError(f"Image API error {response.status_code}: {response.text}")
        result = response.json()
        image_base64 = result.get("data", [{}])[0].get("b64_json")
        image_url = result.get("data", [{}])[0].get("url")
        if not image_base64 and image_url:
            image_response = requests.get(image_url, timeout=60)
            image_response.raise_for_status()
            image_base64 = base64.b64encode(image_response.content).decode("utf-8")
        if not image_base64:
            raise RuntimeError("No image data returned by API")
        return base64.b64decode(image_base64)
