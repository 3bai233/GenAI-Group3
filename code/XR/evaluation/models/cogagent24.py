import os
import re
import types

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class CogAgent24Model():
    def load_model(self, model_name_or_path="path/to/model", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device).eval()

        if not hasattr(self.model.config, "num_hidden_layers"):
            self.model.config.num_hidden_layers = self.model.config.num_layers

        def _prepare_cache_for_generation(self_model, generation_config, model_kwargs, *args, **kwargs):
            model_kwargs["past_key_values"] = None
            return model_kwargs

        self.model._prepare_cache_for_generation = types.MethodType(
            _prepare_cache_for_generation, self.model
        )

        def _extract_past_from_model_output(self_model, outputs):
            past_key_values = None
            if hasattr(outputs, "past_key_values"):
                past_key_values = outputs.past_key_values
            elif isinstance(outputs, dict) and "past_key_values" in outputs:
                past_key_values = outputs["past_key_values"]
            return "past_key_values", past_key_values

        self.model._extract_past_from_model_output = types.MethodType(
            _extract_past_from_model_output, self.model
        )

        self.override_generation_config = GenerationConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True
        ).to_dict()

        self.set_generation_config(
            max_new_tokens=256,
            do_sample=False,
        )

        format_dict = {
            "action_op_sensitive": "(Answer in Action-Operation-Sensitive format.)",
            "status_plan_action_op": "(Answer in Status-Plan-Action-Operation format.)",
            "status_action_op_sensitive": "(Answer in Status-Action-Operation-Sensitive format.)",
            "status_action_op": "(Answer in Status-Action-Operation format.)",
            "action_op": "(Answer in Action-Operation format.)",
        }
        self.format_str = format_dict["action_op"]

    def set_generation_config(self, **kwargs):
        self.override_generation_config.update(kwargs)
        self.model.generation_config = GenerationConfig(**self.override_generation_config)

    def ground_only_positive(self, instruction, image):
        if isinstance(image, str):
            image_path = image
            assert os.path.exists(image_path) and os.path.isfile(image_path), "Invalid input image path."
            image = Image.open(image_path).convert('RGB')
        elif image is None:
            raise ValueError("`image` should be provided.")

        history_str = "\nHistory steps: "
        platform_str = "WIN"
        query = f"Task: {instruction}{history_str}\n{platform_str}{self.format_str}"

        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "image": image, "content": query}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.override_generation_config.get("max_new_tokens", 256),
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=False,
                past_key_values=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            outputs = outputs[:, inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"{response}")

        # Extract bounding boxes from the response
        box_pattern = r"box=\[\[?(\d+),(\d+),(\d+),(\d+)\]?\]"
        matches = re.findall(box_pattern, response)
        if matches:
            bbox = [[int(x) / 1000 for x in match] for match in matches][0]
            click_point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        else:
            bbox = None
            click_point = None
            print("No bounding boxes found in the response.")

        result_dict = {
            "result": "positive",
            "bbox": bbox,
            "point": click_point,
            "raw_response": response
        }
        
        return result_dict


def extract_first_bounding_box(text):
    # Regular expression pattern to match the first bounding box in the format [[x0,y0,x1,y1]]
    pattern = r"\[\[(\d+),(\d+),(\d+),(\d+)\]\]"
    
    # Search for the first match in the text with the DOTALL flag to support multi-line text
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        bbox = [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
        return [pos / 1000 for pos in bbox]
    
    return None


def extract_first_point(text):
    # Regular expression pattern to match the first bounding box in the format [[x0,y0,x1,y1]]
    pattern = r"\[\[(\d+),(\d+)\]\]"
    
    # Search for the first match in the text with the DOTALL flag to support multi-line text
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        bbox = [int(match.group(1)), int(match.group(2))]
        return [pos / 1000 for pos in bbox]
    
    return None
