import json
import os
import re
import tempfile
import base64
from io import BytesIO
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig
import torch
import openai
import math

from qwen_vl_utils import process_vision_info
from .prompt import COMPUTER_USE_DOUBAO, MOBILE_USE_DOUBAO, GROUNDING_DOUBAO

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def extract_bbox(s, image_width=None, image_height=None):
    pattern = r"<\|box_start\|\>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|\>"
    matches = re.findall(pattern, s)
    if matches:
        last_match = matches[-1]
        print(f"BBox: ({last_match[0]}, {last_match[1]}), ({last_match[2]}, {last_match[3]})")
        return (int(last_match[0]), int(last_match[1])), (int(last_match[2]), int(last_match[3]))
    
    pattern = r"<\|box_start\|\>\((\d+),(\d+)\)<\|box_end\|\>"
    matches = re.findall(pattern, s)
    if matches:
        last_match = matches[-1]
        x, y = int(last_match[0]), int(last_match[1])
        print(f"Point: ({x}, {y})")
        return (x, y), (x, y)
        
    return None


def pred_2_point(s):
    floats = re.findall(r'-?\d+\.?\d*', s)
    floats = [float(num) for num in floats]
    if len(floats) == 2:
        return floats
    elif len(floats) == 4:
        return [(floats[0]+floats[2])/2, (floats[1]+floats[3])/2]
    else:
        return None


def normalize_coordinates(coordinates, image_width=None, image_height=None):
    if not coordinates:
        return None
        
    if isinstance(coordinates, tuple) and len(coordinates) == 2 and isinstance(coordinates[0], tuple):
        (x1, y1), (x2, y2) = coordinates
        
        if image_width and image_height:
            smart_resize_height, smart_resize_width = smart_resize(image_height, image_width)
            nx1 = x1 / smart_resize_width
            ny1 = y1 / smart_resize_height
            nx2 = x2 / smart_resize_width
            ny2 = y2 / smart_resize_height
        else:
            nx1 = x1 / 1000
            ny1 = y1 / 1000
            nx2 = x2 / 1000
            ny2 = y2 / 1000
            
        nx1 = max(0.0, min(1.0, nx1))
        ny1 = max(0.0, min(1.0, ny1))
        nx2 = max(0.0, min(1.0, nx2))
        ny2 = max(0.0, min(1.0, ny2))
        
        return [nx1, ny1, nx2, ny2]
    
    elif isinstance(coordinates, list):
        if len(coordinates) == 2:
            x, y = coordinates
            
            if image_width and image_height:
                smart_resize_height, smart_resize_width = smart_resize(image_height, image_width)
                nx = x / smart_resize_width
                ny = y / smart_resize_height
            else:
                nx = x / 1000
                ny = y / 1000
                
            nx = max(0.0, min(1.0, nx))
            ny = max(0.0, min(1.0, ny))
            
            return [nx, ny]
        
        elif len(coordinates) == 4:
            x1, y1, x2, y2 = coordinates
            
            if image_width and image_height:
                smart_resize_height, smart_resize_width = smart_resize(image_height, image_width)
                nx1 = x1 / smart_resize_width
                ny1 = y1 / smart_resize_height
                nx2 = x2 / smart_resize_width
                ny2 = y2 / smart_resize_height
            else:
                nx1 = x1 / 1000
                ny1 = y1 / 1000
                nx2 = x2 / 1000
                ny2 = y2 / 1000
                
            nx1 = max(0.0, min(1.0, nx1))
            ny1 = max(0.0, min(1.0, ny1))
            nx2 = max(0.0, min(1.0, nx2))
            ny2 = max(0.0, min(1.0, ny2))
            
            return [nx1, ny1, nx2, ny2]
    
    return None


def extract_point_from_response(response, image_width=None, image_height=None):
    if '<|box_start|>' in response and '<|box_end|>' in response:
        pred_bbox = extract_bbox(response)
        if pred_bbox:
            (x1, y1), (x2, y2) = pred_bbox
            bbox = normalize_coordinates(((x1, y1), (x2, y2)), image_width, image_height)
            if bbox:
                return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    
    point_pattern = r'<point>(.*?)</point>'
    match = re.search(point_pattern, response)
    if match:
        point_str = match.group(1)
        try:
            x, y = map(float, point_str.split())
            return normalize_coordinates([x, y], image_width, image_height)
        except ValueError:
            pass
    
    click_point = pred_2_point(response)
    if click_point:
        return normalize_coordinates(click_point, image_width, image_height)
    
    return None


def image_to_temp_filename(image):
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(temp_file.name)
    print(f"Image saved to temporary file: {temp_file.name}")
    return temp_file.name


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(height, width, factor=IMAGE_FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


class UITarsModel:
    def load_model(self, model_name_or_path="ByteDance-Seed/UI-TARS-1.5-7B", device="cuda"):
        self.device = device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path, 
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

        try:
            self.generation_config = GenerationConfig.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True
            ).to_dict()
            print("Loaded generation config from model")
        except OSError:
            print("No generation_config.json found, using default config")
            self.generation_config = GenerationConfig().to_dict()
        self.set_generation_config(
            max_length=8192,
            max_new_tokens=64,
            do_sample=False,
            temperature=0.0
        )
    
    def set_generation_config(self, **kwargs):
        self.generation_config.update(**kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)
        
    def _generate_response(self, prompt, image):
        temp_file_path = None
        try:
            if not isinstance(image, str):
                temp_file_path = image_to_temp_filename(image)
                image_path = temp_file_path
            else:
                image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.generation_config.get("max_new_tokens", 64),
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            response = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )[0]
            
            print(response)
            
            return response
        
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            torch.cuda.empty_cache()
    
    def ground_only_positive(self, instruction, image):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
            image_width, image_height = image.size
        else:
            image_path = image
            try:
                with Image.open(image_path) as img:
                    image_width, image_height = img.size
            except Exception:
                image_width, image_height = None, None
                
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."

        prompt = GROUNDING_DOUBAO.format(instruction=instruction)
        response = self._generate_response(prompt, image)
        
        result_dict = {
            "result": "positive",
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }
        
        if '<|box_start|>' in response and '<|box_end|>' in response:
            pred_bbox = extract_bbox(response)
            if pred_bbox is not None:
                (x1, y1), (x2, y2) = pred_bbox
                bbox = normalize_coordinates(((x1, y1), (x2, y2)), image_width, image_height)
                click_point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                
                result_dict["bbox"] = bbox
                result_dict["point"] = click_point
        else:
            action_pattern = r'Action:(.*?)$'
            action_match = re.search(action_pattern, response, re.DOTALL)
            
            if action_match:
                action = action_match.group(1).strip()
                if '<|box_start|>' in action:
                    pred_bbox = extract_bbox(action)
                    if pred_bbox is not None:
                        (x1, y1), (x2, y2) = pred_bbox
                        bbox = normalize_coordinates(((x1, y1), (x2, y2)), image_width, image_height)
                        click_point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                        
                        result_dict["bbox"] = bbox
                        result_dict["point"] = click_point
                else:
                    point = extract_point_from_response(action, image_width, image_height)
                    if point:
                        result_dict["point"] = point
        
        if result_dict["point"] is None:
            click_point = pred_2_point(response)
            if click_point:
                normalized_point = normalize_coordinates(click_point, image_width, image_height)
                result_dict["point"] = normalized_point
        
        return result_dict
    
    def ground_allow_negative(self, instruction, image):
        if not isinstance(image, str):
            assert isinstance(image, Image.Image)
            image_path = image_to_temp_filename(image)
            image_width, image_height = image.size
        else:
            image_path = image
            try:
                with Image.open(image_path) as img:
                    image_width, image_height = img.size
            except Exception:
                image_width, image_height = None, None
                
        assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."

        prompt = GROUNDING_DOUBAO.format(instruction=instruction)
        response = self._generate_response(prompt, image)
        
        result_dict = {
            "result": None,
            "format": "x1y1x2y2",
            "raw_response": response,
            "bbox": None,
            "point": None
        }
        
        if '<|box_start|>' in response and '<|box_end|>' in response:
            pred_bbox = extract_bbox(response)
            if pred_bbox is not None:
                (x1, y1), (x2, y2) = pred_bbox
                bbox = normalize_coordinates(((x1, y1), (x2, y2)), image_width, image_height)
                click_point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                
                result_dict["bbox"] = bbox
                result_dict["point"] = click_point
        else:
            action_pattern = r'Action:(.*?)$'
            action_match = re.search(action_pattern, response, re.DOTALL)
            
            if action_match:
                action = action_match.group(1).strip()
                point = extract_point_from_response(action, image_width, image_height)
                if point:
                    result_dict["point"] = point
            else:
                click_point = pred_2_point(response)
                if click_point:
                    normalized_point = normalize_coordinates(click_point, image_width, image_height)
                    result_dict["point"] = normalized_point
        
        if result_dict["bbox"] or result_dict["point"]:
            result_status = "positive"
        elif "Target does not exist".lower() in response.lower():
            result_status = "negative"
        else:
            result_status = "wrong_format"
        
        result_dict["result"] = result_status
        
        return result_dict
