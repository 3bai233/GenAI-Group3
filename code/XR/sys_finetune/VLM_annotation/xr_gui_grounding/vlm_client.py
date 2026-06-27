from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import dashscope

from xr_gui_grounding.models import (
    Annotations,
    ImagePrompt,
    SceneUnderstanding,
    UiCandidates,
)
from xr_gui_grounding.prompts import (
    annotations_prompt,
    image_prompt_prompt,
    scene_understanding_prompt,
    ui_candidates_prompt,
)
from xr_gui_grounding.utils import extract_json_block, load_image_as_data_url


class VlmClient:
    def __init__(self, api_key: Optional[str], model: str, enable_thinking: bool = False) -> None:
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is required")
        self.model = model
        self.enable_thinking = enable_thinking

    def analyze_scene(self, image_path: str) -> SceneUnderstanding:
        prompt = scene_understanding_prompt()
        response = self._call_json(prompt, "Analyze the image.", image_path)
        return SceneUnderstanding.model_validate(response)

    def suggest_ui_apps(self, image_path: str, scene: SceneUnderstanding, max_apps: int) -> UiCandidates:
        prompt = ui_candidates_prompt(max_apps)
        user_text = (
            "Scene understanding:\n" + scene.model_dump_json(indent=2) + "\n" "Propose UI apps."
        )
        response = self._call_json(prompt, user_text, image_path)
        return UiCandidates.model_validate(response)

    def build_image_prompt(
        self,
        image_path: str,
        scene: SceneUnderstanding,
        app_name: str,
        ui_elements: List[str],
        required_items: List[str],
        window_position: dict,
    ) -> ImagePrompt:
        prompt = image_prompt_prompt()
        user_text = (
            "Scene understanding:\n"
            + scene.model_dump_json(indent=2)
            + "\n"
            + f"Target app: {app_name}\n"
            + f"Required items to show: {required_items}\n"
            + f"UI elements to include: {ui_elements}\n"
            + f"Window position: {window_position}\n"
        )
        response = self._call_json(prompt, user_text, image_path)
        return ImagePrompt.model_validate(response)

    def generate_annotations(
        self,
        image_path: str,
        scene: SceneUnderstanding,
        app_name: str,
        instruction_language: str,
        num_annotations: int,
        bilingual: bool,
    ) -> Annotations:
        prompt = annotations_prompt(instruction_language, num_annotations, bilingual)

        # Build a list of object names that are FORBIDDEN in semantic instructions.
        # This makes the constraint explicit and machine-checkable for the VLM.
        forbidden_names = [obj.name for obj in scene.objects]

        # Identify objects held in hand (position == "bottom" or description mentions "held/hand")
        hand_objects = [
            obj for obj in scene.objects
            if "hand" in obj.description.lower() or "held" in obj.description.lower() or obj.position == "bottom"
        ]
        hand_desc = (
            "  - " + "\n  - ".join(
                f"{obj.name} ({obj.position})" for obj in hand_objects
            )
            if hand_objects
            else "  (none detected)"
        )

        user_text = (
            "## Scene Understanding\n"
            + scene.model_dump_json(indent=2)
            + "\n\n"
            + f"## UI App Name\n{app_name}\n\n"
            + "## Objects likely held in hand (key semantic anchors for semantic_instruction)\n"
            + hand_desc + "\n\n"
            + "## FORBIDDEN WORDS for semantic_instruction\n"
            + "The following object names MUST NOT appear in any semantic_instruction. "
            + "Replace them with relational/positional phrases only:\n"
            + "  " + ", ".join(f'"{n}"' for n in forbidden_names) + "\n\n"
            + "## Self-Check Before Output\n"
            + "Before returning JSON, verify each semantic_instruction:\n"
            + "  1. Does it contain any forbidden word from the list above? → If YES, rewrite.\n"
            + "  2. Can the model find the answer by reading only the UI window (without looking at the real-world background)? → If YES, rewrite.\n"
            + "  3. Does the instruction force the model to look at the real-world scene to identify the correct UI element? → If NO, rewrite.\n"
        )
        response = self._call_json(prompt, user_text, image_path)
        return Annotations.model_validate(response)

    def _call_json(self, system_prompt: str, user_text: str, image_path: str) -> Dict[str, Any]:
        content: List[Dict[str, str]] = [
            {"image": load_image_as_data_url(image_path)},
            {"text": user_text},
        ]
        messages = [
            {"role": "system", "content": [{"text": system_prompt}]},
            {"role": "user", "content": content},
        ]
        kwargs = {}
        if self.enable_thinking:
            kwargs["enable_thinking"] = True

        response = dashscope.MultiModalConversation.call(
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            **kwargs,
        )
        text = _extract_text_from_response(response)
        json_text = extract_json_block(text)
        return json.loads(json_text)


def _extract_text_from_response(response: Any) -> str:
    if isinstance(response, dict):
        output = response.get("output", {})
        choices = output.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, list):
                text_parts = [part.get("text", "") for part in content if "text" in part]
                return "".join(text_parts).strip()
            if isinstance(content, str):
                return content.strip()
    return str(response)
