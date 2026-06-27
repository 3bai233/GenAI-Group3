"""
Local model replacement for GPTJudgeTwoImages
Uses finetuned Qwen3-VL-8B model for two-box judgment
"""

import json
import os
import logging
from copy import deepcopy
from typing import Tuple

import torch
from PIL import Image

from .action import BaseAction


class LocalJudgeTwoImages(BaseAction):
    """
    Use finetuned Qwen3-VL model instead of GPT for two-box judgment
    
    Trained on 128K samples with 92:8 ratio (bbox1:bbox2)
    """
    _required_keys_ = ["image", "image1", "image2"]
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize with finetuned model
        
        Args:
            model_path: Path to finetuned checkpoint
            device: Device to run inference on
        """
        super().__init__()
        # 🔧 从环境变量获取GPU（避免与grounding模型冲突）
        import os
        if device is None:
            device = f"cuda:{os.environ.get('LOCAL_JUDGE_GPU', '0')}"
        self.device = device
        self.model_path = model_path
        
        logging.info(f"Loading finetuned Qwen3-VL model from {model_path}")
        logging.info(f"Judge model will use: {device}")
        self._load_model()
        logging.info(f"✓ Local judge model loaded on {device}")
    
    def _resolve_base_model_path(self) -> str:
        """Resolve the base model path used for loading weights/processor."""
        env_override = os.environ.get("LOCAL_JUDGE_BASE_MODEL")
        if env_override:
            logging.info(f"LOCAL_JUDGE_BASE_MODEL override detected: {env_override}")
            return env_override
        
        adapter_config_path = os.path.join(self.model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            try:
                with open(adapter_config_path, "r", encoding="utf-8") as f:
                    adapter_cfg = json.load(f)
                base_model = adapter_cfg.get("base_model_name_or_path")
                if base_model:
                    logging.info(f"Using base model from adapter config: {base_model}")
                    return base_model
            except Exception as exc:
                logging.warning(f"Failed to parse adapter_config.json: {exc}")
        return self.model_path

    @staticmethod
    def _has_processor_assets(path: str) -> bool:
        """Check whether the given path includes processor assets for multimodal inputs."""
        processor_files = [
            "preprocessor_config.json",
            "processor_config.json",
            "image_processor_config.json",
            "vision_config.json"
        ]
        return any(os.path.exists(os.path.join(path, name)) for name in processor_files)

    def _load_model(self):
        """Load finetuned Qwen3-VL model, attach LoRA adapter if present, and prepare processor."""
        from transformers import AutoModelForVision2Seq, AutoProcessor
        from qwen_vl_utils import process_vision_info

        base_model_path = self._resolve_base_model_path()
        logging.info(f"Loading base model from: {base_model_path}")

        # Prepare device map argument compatible with transformers
        device_map_arg = self.device
        if isinstance(device_map_arg, str) and device_map_arg.startswith("cuda"):
            device_map_arg = {"": device_map_arg}

        # Load base Qwen3-VL model
        self.model = AutoModelForVision2Seq.from_pretrained(
            base_model_path,
            dtype=torch.bfloat16,
            device_map=device_map_arg,
            trust_remote_code=True,
            attn_implementation="eager"
        )

        # Apply LoRA adapter if available
        adapter_weights = os.path.join(self.model_path, "adapter_model.safetensors")
        if os.path.exists(adapter_weights):
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "LoRA adapter weights detected but PEFT is not installed. "
                    "Please install `peft` to use the local judge model."
                ) from exc

            logging.info(f"Applying LoRA adapter from: {self.model_path}")
            self.model = PeftModel.from_pretrained(self.model, self.model_path)

        self.model.eval()

        # Determine processor source
        processor_source = (
            self.model_path if self._has_processor_assets(self.model_path) else base_model_path
        )
        if processor_source != self.model_path:
            logging.info(
                f"No processor assets bundled with adapter; falling back to base model processor: {processor_source}"
            )
        else:
            logging.info(f"Loading processor from adapter directory: {processor_source}")

        self.processor = AutoProcessor.from_pretrained(
            processor_source,
            trust_remote_code=True,
            min_pixels=256 * 28 * 28,
            max_pixels=1024 * 28 * 28
        )

        # Ensure the processor supports multimodal inputs
        if getattr(self.processor, "image_processor", None) is None:
            logging.warning(
                "Loaded processor lacks `image_processor`. Reloading processor from base model."
            )
            self.processor = AutoProcessor.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                min_pixels=256 * 28 * 28,
                max_pixels=1024 * 28 * 28
            )

        self.process_vision_info = process_vision_info
    
    def build_judge_prompt(self, user_query: str) -> str:
        """
        Build prompt for judgment (MUST match training format)
        
        🔧 CRITICAL: Use EXACT same format as training data
        """
        prompt = f"""<image>
<image>

You are comparing two images to determine which one better fulfills the user's intent.

Image 1: Shows a GUI element marked with a green box labeled "1" (first grounding result)
Image 2: Shows a GUI element marked with a red box labeled "2" (regrounding after masking)

Your task: Determine which image shows the element that will best fulfill the user's command.

ANALYSIS APPROACH:
1. Examine what GUI element is highlighted in each image
2. Consider which element better matches the user's intent
3. Think about standard GUI patterns and user expectations
4. Choose the image that shows the more appropriate interaction target

KEY PRINCIPLES:
- Focus on the functional purpose of the highlighted elements
- Consider standard UI patterns (buttons for actions, text fields for input, etc.)
- Choose interactive elements over static text/labels
- ELEMENT QUALITY HIERARCHY (best to worst):
   - Icon + Text together (most informative and complete)
   - Complete icon alone (clear visual indicator)
   - Complete text alone (readable label)
   - Multiple elements in one box OR incomplete elements (ambiguous target)

User Command: "{user_query}"

Provide your answer as: 1 or 2"""
        
        return prompt
    
    def parse_answer(self, response_text: str) -> str:
        """
        Parse model output to extract "1" or "2"
        
        Args:
            response_text: Model's raw output
            
        Returns:
            "1" or "2" (defaults to "1" if parsing fails)
        """
        # Clean response
        response = response_text.strip().lower()
        
        # Try direct match
        if response == "1" or response == "1.":
            return "1"
        if response == "2" or response == "2.":
            return "2"
        
        # Try pattern matching
        import re
        
        # Match "answer: 1" or "select: 2" etc.
        match = re.search(r'[:\s]([12])[.\s]?', response)
        if match:
            return match.group(1)
        
        # Match standalone digit
        match = re.search(r'\b([12])\b', response)
        if match:
            return match.group(1)
        
        # Default to "1" if parsing fails
        logging.warning(f"Failed to parse answer from: {response_text[:100]}")
        return "1"
    
    def judge_two_images(
        self, 
        image1: Image.Image, 
        image2: Image.Image, 
        user_query: str
    ) -> Tuple[str, str, str]:
        """
        Judge which image is better using local model
        
        Args:
            image1: First image (green box)
            image2: Second image (red box)
            user_query: User command
            
        Returns:
            (selected_image, reason, response_text)
            - selected_image: "1" or "2"
            - reason: Explanation
            - response_text: Raw model output
        """
        try:
            # Build prompt
            prompt = self.build_judge_prompt(user_query)
            
            # Prepare inputs for Qwen3-VL (HF backend)
            # 参考 web_demo_mm.py 的 HF 后端实现
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image1},
                    {"type": "image", "image": image2},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process vision info
            image_inputs, video_inputs = self.process_vision_info(messages)
            
            # HF processor 调用（正确方式）
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # Move tensors to the target device while keeping BatchEncoding attributes
            inputs = inputs.to(self.device)
            
            # Generate
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=0.0
                )
            
            # Decode
            response_text = self.processor.decode(
                output_ids[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            # Parse answer
            selected = self.parse_answer(response_text)
            reason = f"Local model selected image {selected}"
            
            logging.info(f"LocalJudge: Query='{user_query[:30]}...', Selected={selected}")
            
            return selected, reason, response_text
            
        except Exception as e:
            logging.error(f"Local judge failed: {e}")
            # Fallback to bbox1
            return "1", f"Error: {str(e)}", str(e)
    
    def compute(self, input_dict, model_dict):
        """
        Main compute method (same interface as GPTJudgeTwoImages)
        """
        # Process single input (from DrawDualBoxesSeparate)
        assert len(input_dict) == 1
        input_dict = input_dict[0]
        
        # Get two images
        image1 = input_dict["image1"]
        image2 = input_dict["image2"]
        bbox1 = input_dict.get("bbox1")
        bbox2 = input_dict.get("bbox2")
        user_query = input_dict["user_query"]
        
        logging.info(f"LocalJudgeTwoImages - Starting judgment")
        logging.info(f"User query: {user_query}")
        logging.info(f"Box 1: {bbox1}")
        logging.info(f"Box 2: {bbox2}")
        
        # Call local model
        selected_image, reason, response_text = self.judge_two_images(
            image1, image2, user_query
        )
        
        logging.info(f"Local judgment result: Selected image {selected_image}")
        logging.info(f"Reason: {reason}")
        
        # Build output (same as GPTJudgeTwoImages)
        output_dict = deepcopy(input_dict)
        output_dict["selected_image"] = selected_image
        output_dict["judge_reason"] = reason
        output_dict["judge_response"] = response_text
        output_dict["judge_method"] = "local_qwen3vl"  # Mark as local
        
        # Set final bbox and point
        if selected_image == "1" and bbox1:
            output_dict["bbox_abs"] = bbox1
            output_dict["point_abs"] = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
            output_dict["image"] = image1
        elif selected_image == "2" and bbox2:
            output_dict["bbox_abs"] = bbox2
            output_dict["point_abs"] = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
            output_dict["image"] = image2
        else:
            # Fallback to bbox1
            output_dict["bbox_abs"] = bbox1
            output_dict["point_abs"] = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2) if bbox1 else None
            output_dict["image"] = image1
        
        return output_dict

