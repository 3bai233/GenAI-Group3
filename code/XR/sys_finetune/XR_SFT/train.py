import os
import json
import random
import torch
import argparse
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from PIL import Image
from io import BytesIO
from collections import Counter
import re

import torch.distributed as dist
from datasets import Dataset
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback,
    AutoProcessor,
    AutoTokenizer,
    AutoConfig,
    AutoModelForImageTextToText,
    PreTrainedTokenizer,
)
import importlib
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from swanlab.integration.transformers import SwanLabCallback
import torch.nn as nn
import math


# ==================== Special Token Definition ====================

POINT_X_TOKEN = "<point_x>"
POINT_Y_TOKEN = "<point_y>"
POINT_START_TOKEN = "<point_start>"
POINT_END_TOKEN = "<point_end>"


def add_special_tokens(tokenizer: PreTrainedTokenizer) -> List[str]:
    """Add special tokens for grounding tasks"""
    special_tokens = [POINT_X_TOKEN, POINT_Y_TOKEN, POINT_START_TOKEN, POINT_END_TOKEN]
    tokenizer.add_special_tokens({
        "additional_special_tokens": special_tokens
    })
    return special_tokens


# ==================== Data Processing ====================

SYSTEM_PROMPT = """You are a helpful assistant. The user will give you an instruction, and you MUST left click on the corresponding UI element via tool call. If you are not sure about where to click, guess a most likely one.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse to interact with a computer.\\n* The screen's resolution is 1000x1000.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. \\n* You can only use the left_click action to interact with the computer.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `left_click`: Click the left mouse button with coordinate (x, y).", "enum": ["left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=left_click`.", "type": "array"}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call> Gould XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""


def convert_percent_bbox_to_xyxy(target_bbox):
    """Convert percentage bbox to pixel coordinates [x1, y1, x2, y2]"""
    if target_bbox is None:
        return None
    ow = target_bbox.get("original_width")
    oh = target_bbox.get("original_height")
    x = target_bbox.get("x")
    y = target_bbox.get("y")
    w = target_bbox.get("width")
    h = target_bbox.get("height")
    if None in (ow, oh, x, y, w, h):
        return None
    x1 = x / 100.0 * ow
    y1 = y / 100.0 * oh
    x2 = (x + w) / 100.0 * ow
    y2 = (y + h) / 100.0 * oh
    return [x1, y1, x2, y2]


def bbox_center_to_1000(bbox_xyxy, img_w, img_h):
    """Normalize pixel bbox center to [0, 1000]"""
    cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2.0
    cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2.0
    norm_x = round(cx / img_w * 1000)
    norm_y = round(cy / img_h * 1000)
    return norm_x, norm_y


def build_answer(norm_x, norm_y):
    return (
        f'<tool_call>\n{{"name": "computer_use", "arguments": {{"action": "left_click", "coordinate": [{norm_x}, {norm_y}]}}}}\n</tool_call>'
    )


def load_bbox_annotation_data(
    json_path: str,
    img_base_dir: str,
    language: str = "en",
    use_direct_instruction: bool = False,
) -> List[Dict]:
    """
    Load new bbox annotation format (summary_with_bbox.json)

    Args:
        use_direct_instruction: If True, add direct_instruction_en samples.
    Returns:
        List of valid samples
    """
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    samples = []
    skipped_invalid_bbox = 0
    skipped_img_not_exist = 0
    skipped_no_instruction = 0

    for item in raw_data:
        orig_image_path = item.get("output_orig_image_path", "")
        if not orig_image_path:
            skipped_img_not_exist += 1
            continue

        img_path = os.path.join(img_base_dir, orig_image_path)
        if not os.path.exists(img_path):
            skipped_img_not_exist += 1
            continue

        annotations = item.get("annotations", [])
        for ann in annotations:
            predicted_bbox = ann.get("predicted_bbox", [])

            if len(predicted_bbox) != 4:
                skipped_invalid_bbox += 1
                continue
            if all(v == -1 for v in predicted_bbox):
                skipped_invalid_bbox += 1
                continue

            x1, y1, x2, y2 = predicted_bbox
            norm_x = max(0, min(1000, round((x1 + x2) / 2.0)))
            norm_y = max(0, min(1000, round((y1 + y2) / 2.0)))

            # semantic instruction (required)
            instruction_sem = ann.get("semantic_instruction" if language == "cn" else "semantic_instruction_en", "")
            if not instruction_sem or not instruction_sem.strip():
                skipped_no_instruction += 1
                continue

            samples.append({
                "img_path": img_path,
                "instruction": instruction_sem.strip(),
                "norm_x": norm_x,
                "norm_y": norm_y,
                "bbox": predicted_bbox,
                "source_image": item.get("source_image", ""),
                "inst_type": "semantic",
            })

            # direct instruction (optional)
            if use_direct_instruction:
                instruction_dir = ann.get("direct_instruction_en", "")
                if instruction_dir and instruction_dir.strip():
                    samples.append({
                        "img_path": img_path,
                        "instruction": instruction_dir.strip(),
                        "norm_x": norm_x,
                        "norm_y": norm_y,
                        "bbox": predicted_bbox,
                        "source_image": item.get("source_image", ""),
                        "inst_type": "direct",
                    })

    print(f"Loaded bbox annotations: {json_path}")
    print(f"  Image not found, skipped: {skipped_img_not_exist}")
    print(f"  Invalid bbox, skipped: {skipped_invalid_bbox}")
    print(f"  No instruction, skipped: {skipped_no_instruction}")
    print(f"  Valid samples: {len(samples)}")
    if use_direct_instruction:
        sem_count = sum(1 for s in samples if s.get("inst_type") == "semantic")
        dir_count = sum(1 for s in samples if s.get("inst_type") == "direct")
        print(f"  Breakdown: semantic={sem_count}, direct={dir_count}")

    return samples


def load_and_filter_data(
    json_path: str,
    img_dir: str,
    task_types: Optional[List[str]] = None,
    language: str = "en",
) -> List[Dict]:
    """
    Load and filter JSON annotation data (legacy format, kept for compatibility)

    Args:
        json_path: JSON annotation file path (supports single file or directory).
        img_dir: Image directory.
        task_types: List of task types to keep, None means all.
        language: "en" or "cn".
    Returns:
        List of valid samples.
    """
    raw_items = []
    if os.path.isdir(json_path):
        for fname in os.listdir(json_path):
            if fname.endswith(".json"):
                with open(os.path.join(json_path, fname), "r") as f:
                    raw_items.extend(json.load(f))
    else:
        with open(json_path, "r") as f:
            raw_items = json.load(f)

    samples = []
    skipped_not_ok = 0
    skipped_no_bbox = 0
    skipped_no_img_field = 0
    skipped_img_not_exist = 0
    skipped_no_instruction = 0
    skipped_no_size = 0

    for item in raw_items:
        if not item.get("is_ok", True):
            skipped_not_ok += 1
            continue

        target_bbox = item.get("target_bbox", {})
        bbox_xyxy = convert_percent_bbox_to_xyxy(target_bbox)
        if bbox_xyxy is None:
            skipped_no_bbox += 1
            continue

        img_filename = item.get("image")
        if not img_filename:
            skipped_no_img_field += 1
            continue

        img_path = os.path.join(img_dir, img_filename)
        if not os.path.exists(img_path):
            skipped_img_not_exist += 1
            continue

        choices = item.get("choices") or {}
        task_type = choices.get("task type")

        if task_types is not None and task_type not in task_types:
            continue

        instruction = (
            item.get("instruction_cn") if language == "cn"
            else item.get("instruction_en")
        )
        if not instruction or not instruction.strip():
            skipped_no_instruction += 1
            continue

        ow = target_bbox.get("original_width")
        oh = target_bbox.get("original_height")
        if not ow or not oh:
            skipped_no_size += 1
            continue

        norm_x, norm_y = bbox_center_to_1000(bbox_xyxy, ow, oh)

        samples.append({
            "img_path": img_path,
            "instruction": instruction.strip(),
            "norm_x": norm_x,
            "norm_y": norm_y,
            "task_type": task_type,
            "annotation_id": item.get("annotation_id"),
        })

    # Print skip statistics
    print(f"Total samples: {len(raw_items)}")
    print(f"  is_ok=False, skipped: {skipped_not_ok}")
    print(f"  bbox parse failed, skipped: {skipped_no_bbox}")
    print(f"  no image field, skipped: {skipped_no_img_field}")
    print(f"  image file not exist, skipped: {skipped_img_not_exist}")
    print(f"  no instruction, skipped: {skipped_no_instruction}")
    print(f"  no image size, skipped: {skipped_no_size}")
    print(f"  Valid samples: {len(samples)}")

    return samples


def split_train_test(samples: List[Dict], test_ratio: float = 0.1, seed: int = 42):
    """Split train/test by ratio with stratified sampling support"""
    random.seed(seed)
    random.shuffle(samples)

    has_inst_type = all("inst_type" in s for s in samples)
    if has_inst_type:
        # Stratified split: sample test_ratio from each instruction type independently
        by_type: Dict[str, List[Dict]] = {}
        for s in samples:
            by_type.setdefault(s["inst_type"], []).append(s)

        train_list, val_list = [], []
        for inst_type, type_samples in by_type.items():
            n_val = max(1, round(len(type_samples) * test_ratio))
            n_val = min(n_val, len(type_samples) - 1)
            val_list.extend(type_samples[:n_val])
            train_list.extend(type_samples[n_val:])

        random.shuffle(train_list)
        random.shuffle(val_list)
        return train_list, val_list

    # Fallback: original uniform split
    total = len(samples)
    n_test = max(1, round(total * test_ratio))
    n_test = min(n_test, total - 1)
    n_train = total - n_test
    return samples[n_test:], samples[:n_test]


# New: Optional training mode
COORD_PATTERN = re.compile(r'coordinate":\s*\[(\d+),\s*(\d+)\]')


def parse_prediction(text: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse predicted coordinates (x, y) from model output text"""
    m = COORD_PATTERN.search(text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def compute_accuracy_from_predictions(
    model,
    test_samples: List[Dict],
    tokenizer,
    processor,
    thresholds: List[int] = [10, 20, 30, 40, 50],
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run inference on test set, parse predicted coordinates, compute multi-threshold accuracy.
    Returns:
        dict with per-threshold accuracy and avg/median distance
    """
    model.eval()
    results = []
    correct = {t: 0 for t in thresholds}
    distances = []

    for sample in test_samples:
        try:
            image = Image.open(sample["img_path"]).convert("RGB")
        except Exception:
            continue

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample["instruction"]},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=None, do_resize=True)

        input_ids = torch.tensor(inputs["input_ids"], device=device)
        attention_mask = torch.tensor(inputs["attention_mask"], device=device)
        pixel_values = inputs["pixel_values"].to(device) if isinstance(inputs["pixel_values"], torch.Tensor) else torch.tensor(inputs["pixel_values"], device=device)
        image_grid_thw = torch.tensor(inputs["image_grid_thw"], device=device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw.unsqueeze(0),
                max_new_tokens=50,
                do_sample=False,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        pred_x, pred_y = parse_prediction(generated_text)

        gt_x, gt_y = sample["norm_x"], sample["norm_y"]
        if pred_x is not None and pred_y is not None:
            dist = ((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2) ** 0.5
            distances.append(dist)
            for t in thresholds:
                if dist <= t:
                    correct[t] += 1
        else:
            distances.append(float("inf"))

    total = len(test_samples)
    if total == 0:
        return {}

    metrics = {f"acc@{t}": correct[t] / total for t in thresholds}
    valid_distances = [d for d in distances if d != float("inf")]
    if valid_distances:
        metrics["avg_dist"] = sum(valid_distances) / len(valid_distances)
        metrics["med_dist"] = sorted(valid_distances)[len(valid_distances) // 2]
    else:
        metrics["avg_dist"] = float("inf")
        metrics["med_dist"] = float("inf")
    metrics["num_parsed"] = len(valid_distances)
    metrics["num_total"] = total
    return metrics


SPLIT_MODE_HELP = {
    "full": "Full random split train/test",
    "train_sg_sa_test_sm": "Simple Grounding + Spatial-anchoring for training; Semantic-matching for testing",
    "train_sg_sm_test_sa": "Simple Grounding + Semantic-matching for training; Spatial-anchoring for testing",
}


def split_by_mode(
    samples: List[Dict],
    split_mode: str,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    if split_mode == "full":
        return split_train_test(samples, test_ratio=test_ratio, seed=seed)

    if split_mode == "train_sg_sa_test_sm":
        train_types = {"Simple Grounding", "Spatial-anchoring"}
        test_types = {"Semantic-matching"}
    elif split_mode == "train_sg_sm_test_sa":
        train_types = {"Simple Grounding", "Semantic-matching"}
        test_types = {"Spatial-anchoring"}
    else:
        raise ValueError(f"Unknown split_mode={split_mode}, options: {list(SPLIT_MODE_HELP.keys())}")

    train_list = [s for s in samples if s.get("task_type") in train_types]
    test_list = [s for s in samples if s.get("task_type") in test_types]

    if not train_list:
        raise ValueError(f"Training set is empty for split_mode={split_mode}")
    if not test_list:
        raise ValueError(f"Test set is empty for split_mode={split_mode}")

    rnd = random.Random(seed)
    rnd.shuffle(train_list)
    rnd.shuffle(test_list)
    return train_list, test_list


# ==================== Dataset Construction ====================

def process_func_text(
    example,
    tokenizer,
    processor,
) -> Dict[str, Any]:
    """Text generation version (original causal LM loss)"""
    MAX_LENGTH = 8192

    img_path = example["img_path"]
    instruction = example["instruction"]
    norm_x = example["norm_x"]
    norm_y = example["norm_y"]

    image = Image.open(img_path).convert("RGB")

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        do_resize=True,
    )

    instruction_input_ids = inputs["input_ids"][0]
    instruction_attention_mask = inputs["attention_mask"][0]
    instruction_pixel_values = inputs["pixel_values"]
    instruction_image_grid_thw = inputs["image_grid_thw"][0]

    answer = build_answer(norm_x, norm_y)
    response = tokenizer(answer, add_special_tokens=False)
    response_input_ids = response["input_ids"]
    response_attention_mask = response.get("attention_mask", [1] * len(response_input_ids))

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        if not response_input_ids or response_input_ids[-1] != eos_token_id:
            response_input_ids = response_input_ids + [eos_token_id]
            response_attention_mask = response_attention_mask + [1]

    input_ids = instruction_input_ids + response_input_ids
    attention_mask = instruction_attention_mask + response_attention_mask
    labels = [-100] * len(instruction_input_ids) + response_input_ids

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    if isinstance(instruction_pixel_values, torch.Tensor):
        pv_list = instruction_pixel_values.tolist()
    else:
        pv_list = instruction_pixel_values.tolist() if hasattr(instruction_pixel_values, "tolist") else instruction_pixel_values

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pv_list,
        "image_grid_thw": instruction_image_grid_thw.tolist()
            if isinstance(instruction_image_grid_thw, torch.Tensor)
            else list(instruction_image_grid_thw),
        "norm_x": norm_x,
        "norm_y": norm_y,
    }


def process_func_mse(
    example,
    tokenizer,
    processor,
) -> Dict[str, Any]:
    """MSE Loss version (using special tokens + regression head)"""
    MAX_LENGTH = 8192

    img_path = example["img_path"]
    instruction = example["instruction"]
    norm_x = example["norm_x"]
    norm_y = example["norm_y"]

    image = Image.open(img_path).convert("RGB")

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        do_resize=True,
    )

    instruction_input_ids = inputs["input_ids"][0]
    instruction_attention_mask = inputs["attention_mask"][0]
    instruction_pixel_values = inputs["pixel_values"]
    instruction_image_grid_thw = inputs["image_grid_thw"][0]

    # Construct answer using special tokens
    # Format: <point_start> <point_x> 500 <point_y> 300 <point_end>
    point_start_id = tokenizer.convert_tokens_to_ids(POINT_START_TOKEN)
    point_end_id = tokenizer.convert_tokens_to_ids(POINT_END_TOKEN)

    if point_start_id == tokenizer.unk_token_id or point_end_id == tokenizer.unk_token_id:
        raise ValueError(
            f"Special tokens not found in tokenizer. "
            f"POINT_START={point_start_id}, POINT_END={point_end_id}. "
            f"Please call add_special_tokens() before training."
        )

    # Normalize coordinates to [0, 1] for regression
    # Keep [0, 1000] range but predict as continuous values
    coord_x = norm_x / 1000.0
    coord_y = norm_y / 1000.0

    # Build response sequence: <point_start> <point_x> <coord_x> <point_y> <coord_y> <point_end>
    # Note: We use token id sequence to mark positions, but coordinate values are not tokenized
    # Instead, we mark them in labels to compute MSE

    # Simplified version: use special tokens only
    # <point_start> [pred_x] [pred_y] <point_end>
    response_ids = [point_start_id]

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        response_ids.append(eos_token_id)

    input_ids = instruction_input_ids + response_ids
    attention_mask = instruction_attention_mask + [1] * len(response_ids)

    # labels: instruction part masked, point_start and eos not included in loss
    # MSE loss computed in custom Trainer
    labels = [-100] * len(instruction_input_ids) + [-100] * len(response_ids)

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    if isinstance(instruction_pixel_values, torch.Tensor):
        pv_list = instruction_pixel_values.tolist()
    else:
        pv_list = instruction_pixel_values.tolist() if hasattr(instruction_pixel_values, "tolist") else instruction_pixel_values

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pv_list,
        "image_grid_thw": instruction_image_grid_thw.tolist()
            if isinstance(instruction_image_grid_thw, torch.Tensor)
            else list(instruction_image_grid_thw),
        "norm_x": norm_x,
        "norm_y": norm_y,
        "coord_x": coord_x,
        "coord_y": coord_y,
    }


class ScreenSpotDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_id_tensors = [torch.as_tensor(s["input_ids"], dtype=torch.long) for s in features]
        attention_tensors = [torch.as_tensor(s["attention_mask"], dtype=torch.long) for s in features]
        label_tensors = [torch.as_tensor(s["labels"], dtype=torch.long) for s in features]

        max_length = max(t.size(0) for t in input_id_tensors)
        pad_id = (
            self.tokenizer.pad_token_id
            if getattr(self.tokenizer, "pad_token_id", None) is not None
            else self.tokenizer.eos_token_id
        )
        if pad_id is None:
            raise ValueError("Both pad_token_id and eos_token_id are None, cannot pad.")

        input_ids = torch.full((len(features), max_length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(features), max_length), dtype=torch.long)
        labels = torch.full((len(features), max_length), -100, dtype=torch.long)

        for idx, (ids, attn, lbl) in enumerate(zip(input_id_tensors, attention_tensors, label_tensors)):
            length = ids.size(0)
            input_ids[idx, :length] = ids
            attention_mask[idx, :length] = attn
            labels[idx, :length] = lbl

        pixel_tensors = []
        for s in features:
            pv = s["pixel_values"]
            if not isinstance(pv, torch.Tensor):
                pv = torch.tensor(pv, dtype=torch.float32)
            pixel_tensors.append(pv)
        pixel_values = torch.cat(pixel_tensors, dim=0)

        image_grid_thw = torch.stack(
            [torch.as_tensor(s["image_grid_thw"], dtype=torch.long).view(-1) for s in features],
            dim=0,
        )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

        # MSE loss version needs to keep coordinate info
        if "coord_x" in features[0]:
            batch["coord_x"] = torch.tensor([s["coord_x"] for s in features], dtype=torch.float32)
            batch["coord_y"] = torch.tensor([s["coord_y"] for s in features], dtype=torch.float32)

        return batch


# ==================== Custom Trainer (with MSE Loss support) ====================

class Qwen3VLRegressionTrainer(Trainer):
    """
    Custom Trainer that computes MSE Loss at specified logits positions
    """

    def __init__(
        self,
        loss_type: str = "text",
        mse_weight: float = 1.0,
        lm_weight: float = 0.0,
        **kwargs
    ):
        """
        Args:
            loss_type: "text" (causal LM) or "mse" (regression)
            mse_weight: Weight for MSE loss
            lm_weight: Weight for language modeling loss (can be set > 0 for mixed training in MSE mode)
        """
        super().__init__(**kwargs)
        self.loss_type = loss_type
        self.mse_weight = mse_weight
        self.lm_weight = lm_weight
        self.point_start_id = None
        self.ignore_token_id = -100

        if self.loss_type == "mse":
            # Delay getting point_start_id until after model is loaded
            pass

    def _get_point_start_id(self):
        """Get point_start token id from tokenizer"""
        if self.point_start_id is None:
            tokenizer = self.data_collator.tokenizer
            self.point_start_id = tokenizer.convert_tokens_to_ids(POINT_START_TOKEN)
            if self.point_start_id == tokenizer.unk_token_id:
                raise ValueError(
                    f"Token '{POINT_START_TOKEN}' not found in tokenizer. "
                    f"Please ensure add_special_tokens() was called."
                )
        return self.point_start_id

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss function

        For MSE version:
        - Output regression values at the logits position corresponding to point_start token
        - Use SmoothL1Loss or MSE Loss to compute difference with target coordinates
        """
        if self.loss_type == "text":
            # Original causal LM loss
            return super().compute_loss(model, inputs, return_outputs, **kwargs)

        # MSE Loss version
        point_start_id = self._get_point_start_id()

        # Forward pass
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            labels=inputs["labels"],
        )

        # Get logits
        logits = outputs.logits  # (batch, seq_len, vocab_size)

        # Find position of point_start token
        batch_size = logits.size(0)
        device = logits.device

        # Initialize predicted coordinates
        pred_x = torch.zeros(batch_size, device=device)
        pred_y = torch.zeros(batch_size, device=device)

        # Target coordinates (already normalized to [0, 1] in collator)
        target_x = inputs["coord_x"].to(device)
        target_y = inputs["coord_y"].to(device)

        for i in range(batch_size):
            # Find position of point_start token
            input_ids_i = inputs["input_ids"][i]
            point_start_positions = (input_ids_i == point_start_id).nonzero(as_tuple=True)[0]

            if len(point_start_positions) > 0:
                pos = point_start_positions[0].item()

                # Get logits at this position
                logits_at_pos = logits[i, pos]  # (vocab_size,)

                # Method 1: Use logits directly as regression output
                # Map vocab_size to [0, 1] range
                # Use softmax as weights, weighted average over token ids
                probs = torch.softmax(logits_at_pos, dim=-1)

                # Create normalized token id weights
                vocab_size = logits.size(-1)
                token_ids = torch.arange(vocab_size, device=device).float()
                normalized_ids = token_ids / vocab_size

                # Weighted sum to get predicted value
                pred_value = (probs * normalized_ids).sum()
                pred_x[i] = pred_value

                # Assume x and y are predicted at consecutive positions
                if pos + 1 < logits.size(1):
                    logits_at_pos_y = logits[i, pos + 1]
                    probs_y = torch.softmax(logits_at_pos_y, dim=-1)
                    pred_y[i] = (probs_y * normalized_ids).sum()
                else:
                    # If not enough positions, assume x == y
                    pred_y[i] = pred_x[i]

        # Compute SmoothL1Loss (more robust to outliers than MSE)
        loss_fn = nn.SmoothL1Loss(reduction="mean")
        loss_x = loss_fn(pred_x, target_x)
        loss_y = loss_fn(pred_y, target_y)
        mse_loss = (loss_x + loss_y) / 2.0

        # If language modeling weight is set, also compute LM loss
        if self.lm_weight > 0:
            lm_loss = outputs.loss
            total_loss = self.mse_weight * mse_loss + self.lm_weight * lm_loss
        else:
            total_loss = mse_loss

        if return_outputs:
            return total_loss, outputs
        return total_loss


# ==================== Vision Encoder LoRA ====================

def get_vision_attention_modules(num_blocks: int = 32) -> List[str]:
    """
    Get LoRA-applicable module names in Qwen3-VL Vision Encoder

    Note: These names are relative to the visual submodel path (without visual. prefix),
    because inject_adapter_in_model acts directly on the visual submodel.

    Qwen3-VL Vision Encoder blocks internal structure:
    - blocks.{i}.attn.qkv: QKV projection (Linear)
    - blocks.{i}.attn.proj: Output projection (Linear)
    - blocks.{i}.mlp.linear_fc1: MLP fc1 (Linear)
    - blocks.{i}.mlp.linear_fc2: MLP fc2 (Linear)
    """
    modules = []
    for i in range(num_blocks):
        modules.extend([
            f"blocks.{i}.attn.qkv",
            f"blocks.{i}.attn.proj",
            f"blocks.{i}.mlp.linear_fc1",
            f"blocks.{i}.mlp.linear_fc2",
        ])
    return modules


def apply_vision_lora(model, lora_config: LoraConfig, rank: int = 32):
    """
    Apply LoRA to Qwen3-VL Vision Encoder

    Since standard PEFT LoRA does not support Conv3d (patch_embed),
    we only apply LoRA to Vision Encoder's Attention and MLP layers.

    PEFT-wrapped model path: peft_model.base_model.model.model.visual
    """
    from peft import inject_adapter_in_model, LoraConfig

    # Get vision encoder - need to penetrate PEFT wrapper layers
    visual_model = None
    # Path: peft_model.base_model.model.model.visual (after PEFT wrapping)
    # Path: model.model.visual (original model)
    candidates = [
        lambda m: m.base_model.model.model.visual,  # PEFT PeftModelForCausalLM
        lambda m: m.base_model.model.visual,         # PEFT alternative structure
        lambda m: m.model.visual,                    # Original Qwen3VLForConditionalGeneration
        lambda m: m.visual,                          # Direct access
    ]
    for getter in candidates:
        try:
            visual_model = getter(model)
            if visual_model is not None:
                break
        except AttributeError:
            continue

    if visual_model is None:
        print("Warning: Cannot find Vision Encoder, skipping Vision LoRA")
        return None

    # Get actual number of vision encoder blocks
    num_blocks = len(visual_model.blocks) if hasattr(visual_model, 'blocks') else 32
    print(f"Vision Encoder: {type(visual_model).__name__}, blocks={num_blocks}")

    vision_modules = get_vision_attention_modules(num_blocks=num_blocks)

    # Create Vision Encoder-specific LoRA config
    vision_lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=vision_modules,
        inference_mode=False,
        r=rank,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias="none",
    )

    # Use inject_adapter_in_model to inject LoRA into vision modules
    try:
        inject_adapter_in_model(vision_lora_config, visual_model, adapter_name="vision_lora")
        # Count injected LoRA parameters
        vision_params = sum(
            p.numel() for n, p in visual_model.named_parameters()
            if "lora_" in n and p.requires_grad
        )
        print(f"Vision Encoder LoRA applied: rank={rank}, blocks={num_blocks}, trainable params={vision_params:,}")
        return visual_model
    except Exception as e:
        print(f"Warning: Vision Encoder LoRA application failed: {e}")
        print("Continuing with LLM-only LoRA...")
        return None


# ==================== Main Function ====================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="train_config.json", help="Training config file path")
    return parser.parse_args()


def main():
    load_dotenv()

    # New: Use SWAN_LAB from .env if present, otherwise don't override (let swanlab use local login cache)
    swan_env = os.getenv("SWAN_LAB", "")
    if swan_env:
        os.environ["SWANLAB_API_KEY"] = swan_env

    args = parse_args()

    # Read config
    with open(args.config, "r") as f:
        cfg = json.load(f)

    # ---- Data Config ----
    json_path = cfg["json_path"]
    img_dir = cfg["img_dir"]
    task_types = cfg.get("task_types")
    language = cfg.get("language", "en")
    test_ratio = cfg.get("test_ratio", 0.1)
    seed = cfg.get("seed", 42)
    split_mode = cfg.get("split_mode", "full")

    # New: Data format type
    data_format = cfg.get("data_format", "bbox_annotation")
    # Options: "bbox_annotation" (summary_with_bbox.json) or "screenspot" (legacy format)

    # New: Independent test set config (ScreenSpot Semantic-matching)
    eval_json_path = cfg.get("eval_json_path", "YOUR_EVAL_JSON_PATH")
    eval_img_dir = cfg.get("eval_img_dir", "YOUR_EVAL_IMG_DIR")
    eval_task_types = cfg.get("eval_task_types", ["Semantic-matching"])

    model_id = cfg["model_id"]
    output_dir = cfg["output_dir"]
    lora_rank = cfg.get("lora_rank", 128)
    lora_alpha = cfg.get("lora_alpha", 16)
    lora_dropout = cfg.get("lora_dropout", 0.05)

    # New: Vision Encoder LoRA
    enable_vision_lora = cfg.get("enable_vision_lora", False)

    # Original target_modules
    base_target_modules = cfg.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj"]

    # New: Loss function type
    loss_type = cfg.get("loss_type", "text")
    # Options: "text" (original causal LM loss) or "mse" (special tokens + MSE loss)

    print(f"=== Training Config ===")
    print(f"Data format: {data_format}")
    print(f"Loss type: {loss_type}")
    print(f"Vision Encoder LoRA: {enable_vision_lora}")
    print(f"Split mode: {split_mode}")
    print(f"Independent test set: {eval_json_path}")

    # ---- Load Training Data ----
    if data_format == "bbox_annotation":
        samples = load_bbox_annotation_data(
            json_path=json_path,
            img_base_dir=img_dir,
            language=language,
            use_direct_instruction=cfg.get("use_direct_instruction", False),
        )
    else:
        samples = load_and_filter_data(
            json_path=json_path,
            img_dir=img_dir,
            task_types=task_types,
            language=language,
        )

    if not samples:
        raise ValueError("No valid training samples")

    # Split validation set by ratio
    val_ratio = cfg.get("val_ratio", cfg.get("test_ratio", 0.1))
    train_samples, val_samples = split_train_test(samples, test_ratio=val_ratio, seed=seed)

    # ---- Load Independent Test Set (ScreenSpot Semantic-matching) ----
    test_samples = []
    if eval_json_path and eval_img_dir:
        print(f"\n=== Loading Independent Test Set ===")
        all_eval_samples = load_and_filter_data(
            json_path=eval_json_path,
            img_dir=eval_img_dir,
            task_types=eval_task_types,
            language=language,
        )
        test_samples = all_eval_samples
        print(f"Independent test set (Semantic-matching): {len(test_samples)} samples")

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # ---- Load Model & Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=False, trust_remote_code=True
    )

    # MSE loss version needs to add special tokens
    if loss_type == "mse":
        print(f"Adding special tokens: {[POINT_START_TOKEN, POINT_END_TOKEN]}")
        add_special_tokens(tokenizer)

    # Limit max pixels
    max_pixels = cfg.get("max_pixels", 1280 * 28 * 28)
    min_pixels = cfg.get("min_pixels", 256 * 28 * 28)
    processor = AutoProcessor.from_pretrained(
        model_id,
        use_fast=False,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
    )

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    arch = (config.architectures or [None])[0]
    module_name = f"transformers.models.{config.model_type}.modeling_{config.model_type}"
    module = importlib.import_module(module_name)
    model_cls = getattr(module, arch)
    model = model_cls.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # Resize embedding to accommodate new special tokens
    if loss_type == "mse":
        model.resize_token_embeddings(len(tokenizer))

    model.config.use_cache = False

    # ---- LoRA Config ----
    # First apply LoRA to LLM part
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=base_target_modules,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)

    # If Vision Encoder LoRA is enabled, apply separately
    if enable_vision_lora:
        print(f"Enabling Vision Encoder LoRA (rank={lora_rank})")
        vision_lora_rank = min(lora_rank, 32)  # Vision LoRA uses smaller rank
        apply_vision_lora(peft_model, lora_config, rank=vision_lora_rank)

    peft_model.enable_input_require_grads()
    peft_model.print_trainable_parameters()

    # ---- Data Processing ----
    # DDP: Each process does its own map (num_proc=None to avoid torchrun fork conflict)
    map_kwargs = {"tokenizer": tokenizer, "processor": processor}

    if loss_type == "mse":
        process_func = process_func_mse
    else:
        process_func = process_func_text

    print("Starting data preprocessing...")

    import pyarrow as pa
    def cast_pixel_values(ds):
        """Force pixel_values to large_list type to avoid Arrow 2GB offset overflow"""
        try:
            ds = ds.cast_column("pixel_values", pa.large_list(pa.float32()))
        except Exception:
            pass  # Skip if already correct type
        return ds

    train_hf = Dataset.from_list(train_samples)
    train_dataset = train_hf.map(
        process_func,
        remove_columns=train_hf.column_names,
        fn_kwargs=map_kwargs,
        num_proc=None,
        writer_batch_size=10,
    )
    train_dataset = cast_pixel_values(train_dataset)

    eval_dataset = None
    test_dataset_mapped = None

    if val_samples:
        val_hf = Dataset.from_list(val_samples)
        print("Processing val_dataset...")
        eval_dataset = val_hf.map(
            process_func,
            remove_columns=val_hf.column_names,
            fn_kwargs=map_kwargs,
            num_proc=None,
            writer_batch_size=10,
        )
        eval_dataset = cast_pixel_values(eval_dataset)
        print(f"val_dataset processed, {len(eval_dataset)} samples")

    if test_samples:
        test_hf = Dataset.from_list(test_samples)
        print("Processing test_dataset (ScreenSpot Semantic-matching)...")
        test_dataset_mapped = test_hf.map(
            process_func,
            remove_columns=test_hf.column_names,
            fn_kwargs=map_kwargs,
            num_proc=None,
            writer_batch_size=10,
        )
        test_dataset_mapped = cast_pixel_values(test_dataset_mapped)
        print(f"test_dataset processed, {len(test_dataset_mapped)} samples")

    # ---- Custom Callback: Independent Evaluation on ScreenSpot Test Set ----
    class ScreenSpotEvalCallback(TrainerCallback):
        """Run inference at specified steps, compute multi-threshold accuracy, print results and write to log file"""

        def __init__(self, test_samples, eval_steps: int, output_dir, tokenizer, processor):
            self.test_samples = test_samples
            self.eval_steps = eval_steps
            self.output_dir = output_dir
            self.tokenizer = tokenizer
            self.processor = processor
            self._logged_this_step = set()

        def on_step_end(self, args, state, control, **kwargs):
            if not self.test_samples:
                return
            if state.global_step <= 0 or state.global_step % self.eval_steps != 0:
                return
            if state.global_step in self._logged_this_step:
                return
            self._logged_this_step.add(state.global_step)

            trainer = kwargs.get("trainer")
            if not trainer:
                return

            is_ddp = dist.is_available() and dist.is_initialized()
            is_main = not is_ddp or dist.get_rank() == 0

            model = trainer.model
            device = next(model.parameters()).device

            try:
                metrics = compute_accuracy_from_predictions(
                    model, self.test_samples, self.tokenizer, self.processor,
                    thresholds=[10, 20, 30, 40, 50], device=device,
                )
            except Exception as e:
                print(f"[Step {state.global_step}] Inference evaluation failed: {e}")
                return

            log_lines = [
                f"[Step {state.global_step}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"  acc@10: {metrics.get('acc@10', 0):.4f}  acc@20: {metrics.get('acc@20', 0):.4f}  acc@30: {metrics.get('acc@30', 0):.4f}  acc@40: {metrics.get('acc@40', 0):.4f}  acc@50: {metrics.get('acc@50', 0):.4f}",
                f"  avg_dist: {metrics.get('avg_dist', float('inf')):.2f}  med_dist: {metrics.get('med_dist', float('inf')):.2f}  parsed: {metrics.get('num_parsed', 0)}/{metrics.get('num_total', 0)}",
            ]
            for line in log_lines:
                print(line)

            if is_main:
                log_path = os.path.join(self.output_dir, "eval_log.txt")
                with open(log_path, "a") as f:
                    f.write("\n".join(log_lines) + "\n")

    # ---- Build Callbacks ----
    callbacks = []
    test_eval_steps = cfg.get("test_eval_steps", 500)
    if test_samples:
        callbacks.append(ScreenSpotEvalCallback(test_samples, eval_steps=test_eval_steps, output_dir=output_dir, tokenizer=tokenizer, processor=processor))

    # ---- SwanLab Callback (only enabled on main process) ----
    is_main = (
        not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
    )
    # Since you've logged in with swanlab login, we don't strictly check API Key length,
    # just inject SwanLabCallback on main process to let it auto-read local credentials
    if is_main:
        swanlab_callback = SwanLabCallback(
            project=cfg.get("swanlab_project", "EgoXR-finetune"),
            workspace=cfg.get("swanlab_workspace", "YOUR_WORKSPACE"),
            experiment_name=cfg.get("swanlab_experiment", "qwen-vl-screenspot"),
            config={
                "model_id": model_id,
                "data_format": data_format,
                "language": language,
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
                "test_samples": len(test_samples),
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "llm_target_modules": base_target_modules,
                "enable_vision_lora": enable_vision_lora,
                "loss_type": loss_type,
                "learning_rate": cfg.get("learning_rate", 1e-4),
                "num_train_epochs": cfg.get("num_train_epochs", 3),
                "per_device_train_batch_size": cfg.get("per_device_train_batch_size", 2),
                "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps", 8),
                "split_mode": split_mode,
            },
        )
        callbacks.append(swanlab_callback)
    else:
        print("SwanLab callback skipped (non-main process or invalid API Key)")

    # ---- TrainingArguments ----
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        num_train_epochs=cfg.get("num_train_epochs", 3),
        learning_rate=cfg.get("learning_rate", 1e-4),
        warmup_steps=cfg.get("warmup_steps", 0),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=cfg.get("logging_steps", 10),
        logging_first_step=True,
        save_steps=cfg.get("save_steps", 500),
        save_total_limit=cfg.get("save_total_limit", 3),
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=cfg.get("eval_steps", 500) if eval_dataset else None,
        load_best_model_at_end=True if eval_dataset else False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        dataloader_num_workers=cfg.get("dataloader_num_workers", 4),
        report_to=cfg.get("report_to", "none"),
        seed=seed,
    )

    print("Initializing Trainer...")

    # Select Trainer based on loss_type
    if loss_type == "mse":
        trainer = Qwen3VLRegressionTrainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=ScreenSpotDataCollator(tokenizer=tokenizer),
            callbacks=callbacks,
            loss_type=loss_type,
            mse_weight=1.0,
            lm_weight=cfg.get("lm_weight", 0.0),
        )
    else:
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=ScreenSpotDataCollator(tokenizer=tokenizer),
            callbacks=callbacks,
        )

    print("Starting training...")
    trainer.train()

    # ---- In DDP mode, only save on main process ----
    is_main_process = (
        not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
    )

    if is_main_process:
        print("Main process: Saving model...")

        # ---- Save ----
        os.makedirs(output_dir, exist_ok=True)

        # Merge LoRA weights
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)

        # ---- Loss Curve ----
        logs = trainer.state.log_history
        steps = [log["step"] for log in logs if "loss" in log]
        losses = [log["loss"] for log in logs if "loss" in log]
        if steps:
            plt.figure()
            plt.plot(steps, losses)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title(f"Training Loss ({loss_type} loss)")
            plt.savefig(os.path.join(output_dir, "training_loss.png"))
            plt.close()

        print("Training complete! Model saved to:", output_dir)
    else:
        print("Non-main process: Skipping save")

    # Synchronize all processes
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        if dist.get_rank() == 0:
            print("All processes synchronized")


if __name__ == "__main__":
    main()
