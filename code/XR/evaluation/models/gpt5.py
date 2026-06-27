import os
import re

import base64
from io import BytesIO
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import openai
from openai import BadRequestError

model_name = "gpt-5-mini"
BASE_URL = os.environ.get("OPENAI_API_BASE", None)
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")


def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


class GPT5Model:
    def __init__(self, model_name=None):
        if model_name is None:
            model_name = os.environ.get("OPENAI_MODEL")
        if BASE_URL is not None:
            self.client = openai.OpenAI(
                api_key=OPENAI_KEY,
                base_url=BASE_URL,
            )
        else:
            self.client = openai.OpenAI(
                api_key=OPENAI_KEY,
            )
        self.model_name = model_name
        self.override_generation_config = {"temperature": 0.0}

    def load_model(self):
        pass

    def set_generation_config(self, **kwargs):
        self.override_generation_config.update(kwargs)

    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert("RGB")
        assert isinstance(image, Image.Image), "Invalid input image."

        base64_image = convert_pil_image_to_base64(image)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are an expert in using electronic devices and interacting with graphic interfaces. You should not call any external tools."}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                },
                            },
                            {
                                "type": "text",
                                "text": "You are asked to find the bounding box of an UI element in the given screenshot corresponding to a given instruction.\n"
                                "Don't output any analysis. Output your result in the format of [[x0,y0,x1,y1]], with x and y ranging from 0 to 1."
                                "The instruction is:\n"
                                f"{instruction}\n",
                            },
                        ],
                    },
                ],
                temperature=self.override_generation_config["temperature"],
                max_tokens=16384,
                reasoning_effort="high",
            )
            response_text = response.choices[0].message.content
        except BadRequestError as e:
            print("OpenAI BadRequestError:", e)
            return {"result": "failed", "bbox": None, "point": None, "raw_response": None, "error": str(e)}

        print("------")
        print(response_text)
        bbox = extract_first_bounding_box(response_text)
        click_point = extract_first_point(response_text)

        if not click_point and bbox:
            click_point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

        result_dict = {"result": "positive", "bbox": bbox, "point": click_point, "raw_response": response_text}

        return result_dict

    def ground_allow_negative(self, instruction, image=None):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert("RGB")
        assert isinstance(image, Image.Image), "Invalid input image."

        base64_image = convert_pil_image_to_base64(image)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are an expert in using electronic devices and interacting with graphic interfaces. You should not call any external tools."}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                },
                            },
                            {
                                "type": "text",
                                "text": "You are asked to find the bounding box of an UI element in the given screenshot corresponding to a given instruction.\n"
                                "Don't output any analysis. Output your result in the format of [[x0,y0,x1,y1]], with x and y ranging from 0 to 1. \n"
                                "If such element does not exist, output only the text 'Target not existent'.\n"
                                "The instruction is:\n"
                                f"{instruction}\n",
                            },
                        ],
                    },
                ],
                temperature=self.override_generation_config["temperature"],
                max_tokens=2048,
            )
            response_text = response.choices[0].message.content
        except BadRequestError as e:
            print("OpenAI BadRequestError:", e)
            return {"result": "failed"}

        # Extract bounding box
        # print("------")
        # print(grounding_prompt)
        print("------")
        print(response_text)
        # print("------")

        if "not existent" in response_text.lower():
            return {"result": "negative", "bbox": None, "point": None, "raw_response": response_text}

        # Try getting groundings
        bbox = extract_first_bounding_box(response_text)
        click_point = extract_first_point(response_text)

        if not click_point and bbox:
            click_point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

        result_dict = {"result": "positive" if bbox or click_point else "negative", "bbox": bbox, "point": click_point, "raw_response": response_text}

        return result_dict

    def ground_with_uncertainty(self, instruction, image=None):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert("RGB")
        assert isinstance(image, Image.Image), "Invalid input image."

        base64_image = convert_pil_image_to_base64(image)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are an expert in using electronic devices and interacting with graphic interfaces. You should not call any external tools."}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                },
                            },
                            {
                                "type": "text",
                                "text": "You are asked to find the bounding box of an UI element in the given screenshot corresponding to a given instruction.\n"
                                "- If such element does not exist in the screenshot, output only the text 'Target not existent'."
                                "- If you are sure such element exists and you are confident in finding it, output your result in the format of [[x0,y0,x1,y1]], with x and y ranging from 0 to 1. \n"
                                "Please find out the bounding box of the UI element corresponding to the following instruction: \n"
                                "The instruction is:\n"
                                f"{instruction}\n",
                            },
                        ],
                    },
                ],
                temperature=self.override_generation_config["temperature"],
                max_tokens=2048,
            )
            response_text = response.choices[0].message.content
        except BadRequestError as e:
            print("OpenAI BadRequestError:", e)
            return {"result": "failed"}

        # Extract bounding box
        # print("------")
        # print(grounding_prompt)
        print("------")
        print(response_text)
        # print("------")

        if "not found" in response_text.lower():
            return {"result": "negative", "bbox": None, "point": None, "raw_response": response_text}

        # Try getting groundings
        bbox = extract_first_bounding_box(response_text)
        click_point = extract_first_point(response_text)

        if not click_point and bbox:
            click_point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

        result_dict = {"result": "positive", "bbox": bbox, "point": click_point, "raw_response": response_text}

        return result_dict


def extract_first_bounding_box(text):
    pattern = r"\[\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]\]"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        bbox = [float(match.group(i)) for i in range(1, 5)]
        if any(v > 1.0 for v in bbox):
            bbox = [v / 1000.0 for v in bbox]
        return bbox
    return None


def extract_first_point(text):
    pattern = r"\[\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]\]"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        point = [float(match.group(1)), float(match.group(2))]
        if any(v > 1.0 for v in point):
            point = [v / 1000.0 for v in point]
        return point
    return None


class GeminiModel(GPT5Model):
    def __init__(self, model_name="gemini-3-flash-preview", base_url=None, api_key=None):
        if model_name is None:
            model_name = os.environ.get("OPENAI_MODEL", "gemini-3-flash-preview")
        base_url = base_url or os.environ.get("OPENAI_API_BASE", "YOUR_URL")
        api_key = api_key or OPENAI_KEY

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model_name = model_name
        self.override_generation_config = {"temperature": 0.0}


class ClaudeModel(GPT5Model):
    def __init__(self, model_name="claude-3.5-sonnet", base_url=None, api_key=None):
        if model_name is None:
            model_name = os.environ.get("OPENAI_MODEL", "claude-3.5-sonnet")
        base_url = base_url or os.environ.get("OPENAI_API_BASE", "YOUR_URL")
        api_key = api_key or OPENAI_KEY

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model_name = model_name
        self.override_generation_config = {"temperature": 0.0}
