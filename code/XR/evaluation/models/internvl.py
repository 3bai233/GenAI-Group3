import os
import re

import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, GenerationConfig


# InternVL helpers



def build_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def internvl_preprocess_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternVLModel():
    def __init__(self):
        self.override_generation_config = dict()

    def load_model(self, model_name_or_path="OpenGVLab/InternVL2-8B", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self._hf_model = AutoModel.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        ).eval()

        self.set_generation_config(
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            max_length=None,
        )

    def set_generation_config(self, **kwargs):
        self.override_generation_config.update(kwargs)
        if hasattr(self, '_hf_model'):
            self._hf_model.generation_config = GenerationConfig(**self.override_generation_config)

    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        assert isinstance(image, Image.Image), "Invalid input image."

        target_device = next(self._hf_model.parameters()).device
        pixel_values = internvl_preprocess_image(image, max_num=12).to(torch.bfloat16).to(target_device)

        grounding_prompt = f"<image>\nPlease provide the bounding box coordinate of the UI element this user instruction describes: <ref>{instruction}</ref>. Answer in the format of [[x1, y1, x2, y2]]"

        response = self._hf_model.chat(self.tokenizer, pixel_values, grounding_prompt, self.override_generation_config)  # 用 _hf_model
        print(f'Assistant: {response}')

        # Extract bounding box
        # print("------")
        # print(grounding_prompt)
        # print("------")
        # print(response)
        # print("------")
        # Try getting groundings
        bbox = extract_last_bounding_box(response)
        click_point = extract_first_point(response)

        if not click_point and bbox:
            click_point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

        result_dict = {
            "result": "positive",
            "bbox": bbox,
            "point": click_point,
            "raw_response": response
        }

        return result_dict


def extract_last_bounding_box(text):
    pattern = r'\[\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\]'
    matches = re.findall(pattern, text)
    print(f"[DEBUG] extract_last_bounding_box: raw_text='{text}' matches={matches}")
    if matches:
        match = matches[-1]
        bbox = [int(match[0]), int(match[1]), int(match[2]), int(match[3])]
        normalized = [pos / 1000 for pos in bbox]
        print(f"[DEBUG] bbox raw={bbox} normalized={normalized}")
        return normalized
    return None

def extract_first_point(text):
    pattern = r"\[\[\s*(\d+)\s*,\s*(\d+)\s*\]\]"
    match = re.search(pattern, text, re.DOTALL)
    print(f"[DEBUG] extract_first_point: raw_text='{text}' match={match}")
    if match:
        point = [int(match.group(1)), int(match.group(2))]
        normalized = [pos / 1000 for pos in point]
        print(f"[DEBUG] point raw={point} normalized={normalized}")
        return normalized
    return None
