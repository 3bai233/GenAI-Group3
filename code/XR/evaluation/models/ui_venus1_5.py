import os
import re
import ast
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info


class UIVenus15Model:
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.generation_config = {
            "max_new_tokens": 256,
            "do_sample": False,
            "temperature": 0.0,
        }

    def load_model(self, model_name_or_path: str):
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto",
            attn_implementation="flash_attention_2",
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

    def set_generation_config(self, **kwargs):
        self.generation_config.update(kwargs)

    def _build_prompt(self, instruction: str, allow_refusal: bool = False):
        instruction = instruction[:-1] if instruction.endswith(".") else instruction
        if allow_refusal:
            return (
                "Output the center point of the position corresponding to the following instruction:\n"
                f"{instruction}.\n\n"
                "The output should just be the coordinates of a point, in the format [x,y]. "
                "Additionally, if the task is infeasible (e.g., the task is not related to the image), "
                "the output should be [-1,-1]."
            )
        return (
            "Output the center point of the position corresponding to the following instruction:\n"
            f"{instruction}.\n\n"
            "The output should just be the coordinates of a point, in the format [x,y]."
        )

    def _parse_point(self, text: str):
        text = text.strip()

        pattern_bbox = r"\[\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*\]"
        pattern_point = r"\[\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*\]"
        pattern_two_points = r"\[\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*\]\s*,\s*\[\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*\]"

        try:
            if re.fullmatch(pattern_bbox, text, re.DOTALL):
                box = ast.literal_eval(text)
                x = (float(box[0]) + float(box[2])) / 2.0
                y = (float(box[1]) + float(box[3])) / 2.0
            elif re.fullmatch(pattern_point, text, re.DOTALL):
                pt = ast.literal_eval(text)
                x, y = float(pt[0]), float(pt[1])
            elif re.fullmatch(pattern_two_points, text.replace(" ", ""), re.DOTALL):
                pts = ast.literal_eval("[" + text + "]")
                x = (float(pts[0][0]) + float(pts[1][0])) / 2.0
                y = (float(pts[0][1]) + float(pts[1][1])) / 2.0
            else:
                m = re.search(r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]", text)
                if not m:
                    return None
                x, y = float(m.group(1)), float(m.group(2))

            if x == -1 and y == -1:
                return None

            # UI-Venus脚本默认1000网格
            return [x / 1000.0, y / 1000.0]
        except Exception:
            return None

    def ground_only_positive(self, instruction: str, image):
        if isinstance(image, str):
            if not os.path.exists(image):
                return {"point": None, "raw_response": f"image not found: {image}"}
            image_input = image  # 给 process_vision_info 用路径
        elif isinstance(image, Image.Image):
            image_input = image
        else:
            return {"point": None, "raw_response": "invalid image input"}

        prompt = self._build_prompt(instruction, allow_refusal=False)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_input},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": self.generation_config.get("max_new_tokens", 256),
            "do_sample": self.generation_config.get("do_sample", False),
        }
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = self.generation_config.get("temperature", 0.0)

        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        point = self._parse_point(output_text)
        return {"point": point, "raw_response": output_text}

    def ground_allow_negative(self, instruction: str, image):
        # 为兼容 eval_screenspot_pro.py negative 分支
        out = self.ground_only_positive(instruction, image)
        if out["point"] is None:
            return {"result": "negative", "point": None, "raw_response": out["raw_response"]}
        return {"result": "positive", "point": out["point"], "raw_response": out["raw_response"]}