from __future__ import annotations

import ast
import base64
import json
import logging
import os
import sys
from copy import deepcopy
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image, ImageDraw, ImageFont

from mm_agents.uitars15_v2 import (
    ENV_FAIL_WORD,
    FINISH_WORD,
    WAIT_WORD,
    UITarsAgent,
    logger,
    parse_action_to_structure_output,
    parsing_response_to_pyautogui_code,
)


class UITarsBamiAgent(UITarsAgent):
    def __init__(
        self,
        model: str,
        model_type: str,
        max_tokens: int,
        top_p: Optional[float],
        temperature: float,
        max_trajectory_length: Optional[int],
        max_image_history_length: Optional[int],
        screenshot_pyautogui_prompt: str = "uitars_v1",
        which_parsed_actions: str = "all",
        max_steps: int = 100,
        use_thinking: bool = True,
        language: str = "Chinese",
        enable_bami: bool = False,
        bami_local_judge_model_path: Optional[str] = None,
        bami_local_judge_base_model_path: Optional[str] = None,
        bami_local_judge_gpu: Optional[str] = None,
        bami_mask_ratio: float = 0.12,
        bami_crop_expand_ratio: float = 0.2,
    ):
        super().__init__(
            model=model,
            model_type=model_type,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            max_trajectory_length=max_trajectory_length,
            max_image_history_length=max_image_history_length,
            screenshot_pyautogui_prompt=screenshot_pyautogui_prompt,
            which_parsed_actions=which_parsed_actions,
            max_steps=max_steps,
            use_thinking=use_thinking,
            language=language,
        )
        self.enable_bami = enable_bami
        self.bami_local_judge_model_path = bami_local_judge_model_path
        self.bami_local_judge_base_model_path = bami_local_judge_base_model_path
        self.bami_local_judge_gpu = bami_local_judge_gpu
        self.bami_mask_ratio = bami_mask_ratio
        self.bami_crop_expand_ratio = bami_crop_expand_ratio
        self._bami_local_judge = None
        if self.enable_bami:
            self._bami_local_judge = self._try_create_local_judge()

    def _feedcog_root(self) -> str:
        env_override = os.environ.get("FEEDCOG_ROOT")
        if env_override and os.path.isdir(env_override):
            return env_override

        search_roots = [os.path.abspath(os.path.dirname(__file__)), os.path.abspath(os.getcwd())]
        checked = set()
        for search_root in search_roots:
            current = search_root
            while current and current not in checked:
                checked.add(current)
                candidate = os.path.join(current, "FeedCoG")
                if os.path.isdir(candidate):
                    return candidate
                parent = os.path.dirname(current)
                if parent == current:
                    break
                current = parent

        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "FeedCoG")
        )

    def _resolve_local_judge_base_model_path(self) -> Optional[str]:
        if self.bami_local_judge_base_model_path:
            return self.bami_local_judge_base_model_path

        env_override = os.environ.get("LOCAL_JUDGE_BASE_MODEL")
        if env_override:
            return env_override

        if not self.bami_local_judge_model_path:
            return None

        adapter_config_path = os.path.join(self.bami_local_judge_model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            try:
                with open(adapter_config_path, "r", encoding="utf-8") as handle:
                    adapter_config = json.load(handle)
                base_model_name = adapter_config.get("base_model_name_or_path")
            except Exception as exc:
                self.logger.warning(f"Failed to parse local judge adapter config: {exc}")
                base_model_name = None

            if base_model_name:
                candidate_paths = [
                    base_model_name,
                    os.path.join("/data/models", base_model_name),
                    os.path.join("/share/home/group3/agent/OSWorld", os.path.basename(base_model_name)),
                ]
                for candidate_path in candidate_paths:
                    if os.path.isdir(candidate_path):
                        return candidate_path

                self.logger.warning(
                    "Local judge adapter expects base model %s, but no local directory was found.",
                    base_model_name,
                )

        return None

    def _try_create_local_judge(self):
        if not self.bami_local_judge_model_path:
            self.logger.warning(
                "BAMI enabled but no local judge model path provided; refinement will fall back to baseline."
            )
            return None

        feedcog_root = self._feedcog_root()
        if feedcog_root not in sys.path:
            sys.path.insert(0, feedcog_root)

        try:
            from utils.action_local_judge import LocalJudgeTwoImages
        except Exception as exc:
            self.logger.warning("Failed to import FeedCoG LocalJudgeTwoImages: %s", exc)
            return None

        previous_gpu = os.environ.get("LOCAL_JUDGE_GPU")
        previous_base_model = os.environ.get("LOCAL_JUDGE_BASE_MODEL")
        resolved_base_model_path = self._resolve_local_judge_base_model_path()
        try:
            if self.bami_local_judge_gpu is not None:
                os.environ["LOCAL_JUDGE_GPU"] = str(self.bami_local_judge_gpu)
            if resolved_base_model_path:
                os.environ["LOCAL_JUDGE_BASE_MODEL"] = resolved_base_model_path
            return LocalJudgeTwoImages(model_path=self.bami_local_judge_model_path)
        except Exception as exc:
            self.logger.warning(f"Failed to initialize BAMI local judge: {exc}")
            return None
        finally:
            if self.bami_local_judge_gpu is not None:
                if previous_gpu is None:
                    os.environ.pop("LOCAL_JUDGE_GPU", None)
                else:
                    os.environ["LOCAL_JUDGE_GPU"] = previous_gpu
            if resolved_base_model_path:
                if previous_base_model is None:
                    os.environ.pop("LOCAL_JUDGE_BASE_MODEL", None)
                else:
                    os.environ["LOCAL_JUDGE_BASE_MODEL"] = previous_base_model

    def _screenshot_to_base64(self, screenshot: Union[bytes, str]) -> str:
        if isinstance(screenshot, bytes):
            return base64.b64encode(screenshot).decode("utf-8")
        return screenshot

    def _build_messages(self, task_instruction: str, current_screenshot_b64: str) -> List[Dict]:
        messages = [{"role": "user", "content": [{"type": "text", "text": self.system_prompt.format(instruction=task_instruction, language=self.language)}]}]
        history_images = list(self.history_images)
        required_images = len(self.history_responses) + 1
        if len(history_images) < required_images:
            history_images.append(current_screenshot_b64)
        elif history_images:
            history_images[-1] = current_screenshot_b64
        else:
            history_images = [current_screenshot_b64]
        image_num = 0
        if len(self.history_responses) > 0:
            for history_idx, history_response in enumerate(self.history_responses):
                if history_idx + self.history_n > len(self.history_responses):
                    messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{history_images[image_num]}"}}]})
                    image_num += 1
                messages.append({"role": "assistant", "content": history_response})
            messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{history_images[image_num]}"}}]})
        else:
            messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{history_images[image_num]}"}}]})
        return messages

    def _sanitize_prediction_text(self, prediction: str) -> str:
        if not isinstance(prediction, str):
            return prediction

        marker = 'Action:'
        if marker not in prediction:
            return prediction

        prefix, action_part = prediction.rsplit(marker, 1)
        stripped_action = action_part.strip()
        broken_prefix = "hotkey(key='"

        if stripped_action.startswith(broken_prefix) and not stripped_action.endswith("')"):
            key_text = stripped_action[len(broken_prefix):]
            key_text = key_text.strip().rstrip(')')
            key_text = key_text.replace(' ', '+')
            if key_text:
                fixed_action = f"hotkey(key='{key_text}')"
                self.logger.warning(
                    f"Sanitized malformed action from model output: {stripped_action} -> {fixed_action}"
                )
                return f"{prefix}{marker} {fixed_action}"

        return prediction

    def _sanitize_prediction_text(self, prediction: str) -> str:
        if not isinstance(prediction, str):
            return prediction

        marker = 'Action:'
        if marker not in prediction:
            return prediction

        prefix, action_part = prediction.rsplit(marker, 1)
        stripped_action = action_part.strip()
        broken_prefix = "hotkey(key='"

        if stripped_action.startswith(broken_prefix) and not stripped_action.endswith("')"):
            key_text = stripped_action[len(broken_prefix):]
            key_text = key_text.strip().rstrip(')')
            key_text = key_text.replace(' ', '+')
            if key_text:
                fixed_action = f"hotkey(key='{key_text}')"
                self.logger.warning(
                    f"Sanitized malformed action from model output: {stripped_action} -> {fixed_action}"
                )
                return f"{prefix}{marker} {fixed_action}"

        return prediction

    def _sanitize_prediction_text(self, prediction: str) -> str:
        if not isinstance(prediction, str):
            return prediction

        marker = 'Action:'
        if marker not in prediction:
            return prediction

        prefix, action_part = prediction.rsplit(marker, 1)
        stripped_action = action_part.strip()
        broken_prefix = "hotkey(key='"

        if stripped_action.startswith(broken_prefix) and not stripped_action.endswith("')"):
            key_text = stripped_action[len(broken_prefix):]
            key_text = key_text.strip().rstrip(')')
            key_text = key_text.replace(' ', '+')
            if key_text:
                fixed_action = f"hotkey(key='{key_text}')"
                self.logger.warning(
                    f"Sanitized malformed action from model output: {stripped_action} -> {fixed_action}"
                )
                return f"{prefix}{marker} {fixed_action}"

        return prediction

    def _sanitize_prediction_text(self, prediction: str) -> str:
        if not isinstance(prediction, str):
            return prediction

        marker = 'Action:'
        if marker not in prediction:
            return prediction

        prefix, action_part = prediction.rsplit(marker, 1)
        stripped_action = action_part.strip()
        broken_prefix = "hotkey(key='"

        if stripped_action.startswith(broken_prefix) and not stripped_action.endswith("')"):
            key_text = stripped_action[len(broken_prefix):]
            key_text = key_text.strip().rstrip(')')
            key_text = key_text.replace(' ', '+')
            if key_text:
                fixed_action = f"hotkey(key='{key_text}')"
                self.logger.warning(
                    f"Sanitized malformed action from model output: {stripped_action} -> {fixed_action}"
                )
                return f"{prefix}{marker} {fixed_action}"

        return prediction

    def _sanitize_prediction_text(self, prediction: str) -> str:
        if not isinstance(prediction, str):
            return prediction

        marker = 'Action:'
        if marker not in prediction:
            return prediction

        prefix, action_part = prediction.rsplit(marker, 1)
        stripped_action = action_part.strip()
        broken_prefix = "hotkey(key='"

        if stripped_action.startswith(broken_prefix) and not stripped_action.endswith("')"):
            key_text = stripped_action[len(broken_prefix):]
            key_text = key_text.strip().rstrip(')')
            key_text = key_text.replace(' ', '+')
            if key_text:
                fixed_action = f"hotkey(key='{key_text}')"
                self.logger.warning(
                    f"Sanitized malformed action from model output: {stripped_action} -> {fixed_action}"
                )
                return f"{prefix}{marker} {fixed_action}"

        return prediction

    def _sanitize_prediction_text(self, prediction: str) -> str:
        if not isinstance(prediction, str):
            return prediction

        marker = 'Action:'
        if marker not in prediction:
            return prediction

        prefix, action_part = prediction.rsplit(marker, 1)
        stripped_action = action_part.strip()
        broken_prefix = "hotkey(key='"

        if stripped_action.startswith(broken_prefix) and not stripped_action.endswith("')"):
            key_text = stripped_action[len(broken_prefix):]
            key_text = key_text.strip().rstrip(')')
            key_text = key_text.replace(' ', '+')
            if key_text:
                fixed_action = f"hotkey(key='{key_text}')"
                self.logger.warning(
                    f"Sanitized malformed action from model output: {stripped_action} -> {fixed_action}"
                )
                return f"{prefix}{marker} {fixed_action}"

        return prediction

    def _sanitize_prediction_text(self, prediction: str) -> str:
        if not isinstance(prediction, str):
            return prediction

        marker = 'Action:'
        if marker not in prediction:
            return prediction

        prefix, action_part = prediction.rsplit(marker, 1)
        stripped_action = action_part.strip()
        broken_prefix = "hotkey(key='"

        if stripped_action.startswith(broken_prefix) and not stripped_action.endswith("')"):
            key_text = stripped_action[len(broken_prefix):]
            key_text = key_text.strip().rstrip(')')
            key_text = key_text.replace(' ', '+')
            if key_text:
                fixed_action = f"hotkey(key='{key_text}')"
                self.logger.warning(
                    f"Sanitized malformed action from model output: {stripped_action} -> {fixed_action}"
                )
                return f"{prefix}{marker} {fixed_action}"

        return prediction

    def _run_prediction_once(self, messages: List[Dict], origin_resized_height: int, origin_resized_width: int) -> Tuple[Optional[str], Optional[List[Dict]], Optional[str]]:
        try:
            prediction = self.inference_func(messages)
        except Exception as exc:
            self.logger.error(f"Error when fetching response from client: {exc}")
            return None, None, None
        sanitized_prediction = self._sanitize_prediction_text(prediction)
        try:
            parsed_dict = parse_action_to_structure_output(sanitized_prediction, self.action_parse_res_factor, origin_resized_height, origin_resized_width, self.model_type)
            parsed_pyautogui_code = parsing_response_to_pyautogui_code(parsed_dict, origin_resized_height, origin_resized_width)
            return sanitized_prediction, parsed_dict, parsed_pyautogui_code
        except Exception as exc:
            self.logger.error(f"Error when parsing response from client: {exc}")
            return sanitized_prediction, None, None

    def _extract_action_box(self, parsed_response: Dict, image_width: int, image_height: int) -> Optional[Tuple[str, List[int]]]:
        action_inputs = parsed_response.get("action_inputs", {})
        box_key = None
        box_text = None
        for candidate_key in ("start_box", "point"):
            candidate_value = action_inputs.get(candidate_key)
            if candidate_value:
                box_key = candidate_key
                box_text = candidate_value
                break
        if not box_text:
            return None
        try:
            coords = ast.literal_eval(box_text)
        except Exception:
            return None
        if len(coords) == 2:
            coords = [coords[0], coords[1], coords[0], coords[1]]
        if len(coords) != 4:
            return None
        x1, y1, x2, y2 = [float(value) for value in coords]
        abs_bbox = [round(min(x1, x2) * image_width), round(min(y1, y2) * image_height), round(max(x1, x2) * image_width), round(max(y1, y2) * image_height)]
        return box_key, abs_bbox

    def _bbox_to_action_box(self, box_key: str, bbox: List[int], image_width: int, image_height: int) -> str:
        x1, y1, x2, y2 = bbox
        if box_key == "point":
            return str([((x1 + x2) / 2) / image_width, ((y1 + y2) / 2) / image_height])
        return str([x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height])

    def _action_is_refinable(self, parsed_response: Dict) -> bool:
        action_inputs = parsed_response.get("action_inputs", {})
        return any(action_inputs.get(key) for key in ("start_box", "point"))

    def _expanded_mask_bbox(self, image: Image.Image, bbox: List[int]) -> List[int]:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        half_width = max(abs(x2 - x1) / 2, image.width * self.bami_mask_ratio / 2, 2)
        half_height = max(abs(y2 - y1) / 2, image.height * self.bami_mask_ratio / 2, 2)

        return [
            max(0, int(center_x - half_width)),
            max(0, int(center_y - half_height)),
            min(image.width - 1, int(center_x + half_width)),
            min(image.height - 1, int(center_y + half_height)),
        ]

    def _mask_bbox(self, image: Image.Image, bbox: List[int]) -> Image.Image:
        masked = image.copy().convert("RGB")
        ImageDraw.Draw(masked).rectangle(self._expanded_mask_bbox(image, bbox), fill=(0, 0, 0))
        return masked

    def _render_candidate_focus(self, image: Image.Image, bbox: List[int], label: str, color: Tuple[int, int, int]) -> Image.Image:
        marked = image.copy().convert("RGB")
        overlay = Image.new("RGBA", marked.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except Exception:
            font = ImageFont.load_default()
        draw_overlay.rectangle(bbox, outline=color + (255,), width=3)
        draw_overlay.text((bbox[0], max(0, bbox[1] - 30)), label, fill=color + (255,), font=font)
        merged = Image.alpha_composite(marked.convert("RGBA"), overlay).convert("RGB")
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        expand_width = int(merged.width * self.bami_crop_expand_ratio)
        expand_height = int(merged.height * self.bami_crop_expand_ratio)
        crop_left = max(0, int(center_x - expand_width))
        crop_top = max(0, int(center_y - expand_height))
        crop_right = min(merged.width, int(center_x + expand_width))
        crop_bottom = min(merged.height, int(center_y + expand_height))
        return merged.crop((crop_left, crop_top, crop_right, crop_bottom))

    def _judge_candidates(self, image: Image.Image, baseline_bbox: List[int], reground_bbox: List[int], task_instruction: str) -> Tuple[str, str, str]:
        if self._bami_local_judge is None:
            return "1", "BAMI local judge unavailable; keeping baseline.", "judge_unavailable"
        image1 = self._render_candidate_focus(image, baseline_bbox, "1", (0, 255, 0))
        image2 = self._render_candidate_focus(image, reground_bbox, "2", (255, 0, 0))
        return self._bami_local_judge.judge_two_images(image1, image2, task_instruction)

    def _maybe_refine_with_bami(self, task_instruction: str, screenshot_image: Image.Image, baseline_prediction: str, parsed_dict: List[Dict], image_width: int, image_height: int) -> Tuple[Union[str, Dict], List[Dict]]:
        if not self.enable_bami or not parsed_dict:
            return baseline_prediction, parsed_dict
        candidate_index = None
        for idx, parsed_response in enumerate(parsed_dict):
            if self._action_is_refinable(parsed_response):
                candidate_index = idx
                break
        if candidate_index is None:
            return baseline_prediction, parsed_dict
        baseline_action_box = self._extract_action_box(parsed_dict[candidate_index], image_width, image_height)
        if baseline_action_box is None:
            return baseline_prediction, parsed_dict
        box_key, baseline_bbox = baseline_action_box
        masked_image = self._mask_bbox(screenshot_image, baseline_bbox)
        buffer = BytesIO()
        masked_image.save(buffer, format="PNG")
        masked_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        masked_messages = self._build_messages(task_instruction, masked_b64)
        reground_prediction, reground_parsed_dict, _ = self._run_prediction_once(masked_messages, image_height, image_width)
        if not reground_prediction or not reground_parsed_dict:
            return {"raw_prediction": baseline_prediction, "bami": {"status": "reground_parse_failed", "kept": "baseline"}}, parsed_dict
        reground_box_key = None
        reground_bbox = None
        for parsed_response in reground_parsed_dict:
            if self._action_is_refinable(parsed_response):
                reground_action_box = self._extract_action_box(parsed_response, image_width, image_height)
                if reground_action_box is not None:
                    reground_box_key, reground_bbox = reground_action_box
                    break
        if reground_bbox is None:
            return {"raw_prediction": baseline_prediction, "bami": {"status": "reground_no_bbox", "kept": "baseline"}}, parsed_dict
        selected, judge_reason, judge_response = self._judge_candidates(screenshot_image, baseline_bbox, reground_bbox, task_instruction)
        final_parsed_dict = deepcopy(parsed_dict)
        final_source = "baseline"
        if selected == "2":
            target_box_key = box_key
            if reground_box_key is not None:
                target_box_key = reground_box_key if reground_box_key in final_parsed_dict[candidate_index].get("action_inputs", {}) else box_key
            final_parsed_dict[candidate_index]["action_inputs"][target_box_key] = self._bbox_to_action_box(target_box_key, reground_bbox, image_width, image_height)
            final_source = "reground"
        return {"raw_prediction": baseline_prediction, "reground_prediction": reground_prediction, "bami": {"status": "refined", "selected": selected, "final_source": final_source, "action_type": parsed_dict[candidate_index].get("action_type"), "box_key": box_key, "judge_reason": judge_reason, "judge_response": judge_response, "baseline_bbox": baseline_bbox, "reground_bbox": reground_bbox}}, final_parsed_dict

    def predict(self, task_instruction: str, obs: dict) -> Tuple[Union[str, Dict, None], List]:
        self.task_instruction = task_instruction
        assert len(self.observations) == len(self.actions) and len(self.actions) == len(self.thoughts), "The number of observations and actions should be the same."
        screenshot_b64 = self._screenshot_to_base64(obs["screenshot"])
        screenshot_bytes = base64.b64decode(screenshot_b64)
        screenshot_image = Image.open(BytesIO(screenshot_bytes)).convert("RGB")
        image_width, image_height = screenshot_image.size
        self.history_images.append(screenshot_b64)
        self.observations.append({"screenshot": screenshot_b64, "accessibility_tree": None})
        if len(self.history_images) > self.history_n:
            self.history_images = self.history_images[-self.history_n:]
        messages = self._build_messages(task_instruction, screenshot_b64)
        try_times = 3
        prediction = None
        parsed_dict = None
        parsed_pyautogui_code = None
        while True:
            if try_times <= 0:
                self.logger.error("Reach max retry times to fetch response from client, as error flag.")
                return prediction, ["FAIL"]
            self.logger.info(f"Messages: {self.pretty_print_messages(messages[-1])}")
            prediction, parsed_dict, parsed_pyautogui_code = self._run_prediction_once(messages, image_height, image_width)
            if prediction and parsed_dict and parsed_pyautogui_code:
                break
            try_times -= 1
        self.history_responses.append(prediction)
        response_payload, final_parsed_dict = self._maybe_refine_with_bami(task_instruction, screenshot_image, prediction, parsed_dict, image_width, image_height)
        try:
            final_pyautogui_code = parsing_response_to_pyautogui_code(final_parsed_dict, image_height, image_width)
        except Exception as exc:
            self.logger.error("Parsing action error after BAMI refinement: %s", exc)
            return response_payload, ["FAIL"]
        thoughts = ""
        for parsed_response in final_parsed_dict:
            if "thought" in parsed_response and parsed_response["thought"]:
                thoughts += parsed_response["thought"]
        if thoughts:
            self.thoughts.append(thoughts)
        for parsed_response in final_parsed_dict:
            if "action_type" not in parsed_response:
                continue
            if parsed_response["action_type"] == FINISH_WORD:
                self.actions.append(["DONE"])
                return response_payload, ["DONE"]
            if parsed_response["action_type"] == WAIT_WORD:
                self.actions.append(["WAIT"])
                return response_payload, ["WAIT"]
            if parsed_response["action_type"] == ENV_FAIL_WORD:
                self.actions.append(["FAIL"])
                return response_payload, ["FAIL"]
        self.actions.append([final_pyautogui_code])
        return response_payload, [final_pyautogui_code]
