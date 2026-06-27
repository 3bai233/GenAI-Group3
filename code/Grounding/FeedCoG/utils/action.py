from typing import Dict
import re
import logging
import os
import json
from PIL import Image
from copy import deepcopy

from transformers import Qwen2_5_VLProcessor

from .image import mask_image, compute_crop_box
from .generate import generate

def get_qwen3_system_prompt(x, y):
    """Generate qwen3 format system prompt with resolution"""
    system_content = [
        {'text': 'You are a helpful assistant.'},
        {'text': """\n\n# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools> . . . </tools> XML tags:
<tools>{ "name":"computer_use", "description": "Use a mouse to interact with a computer. The screen's resolution is {{screen_width}}x {{screen_height}}." "notes": "Click with the cursor tip centered on targets; avoid edges unless asked. Do not use other tools (type, key, scroll, left_click_drag). Only left_click and mouse_move are allowed. If you can't find the element, terminate and report failure.", "parameters":{ "type":"object", "required":["action"], "properties":{ "action":{ "type":"string", "enum":["mouse_move","left_click"], "description":"The action to perform." }, "coordinate":{ "type":"array", "description":"(x, y): pixels from left/top. Required for action=mouse_move and action=left_click." } } } }
</tools>
For each function call, return a JSON object with function name and arguments within <tool_call> . . . </tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
Additionally, if you think the task is infeasible (e.g., the task is not related to the image), return:
<tool_call>
{"name": "computer_use", "arguments": {"action": "terminate", "status": "failure"}}
</tool_call>"""}
    ]
    system_str = ''.join([item['text'] for item in system_content])
    system_str = system_str.replace("{{screen_width}}", str(x)).replace("{{screen_height}}", str(y))
    return system_str

def get_input(padded_size, user_query, processor: Qwen2_5_VLProcessor, prompt_type="tianxi"):
    """
    Generate input prompt for the model
    
    Args:
        padded_size: (width, height) tuple
        user_query: user instruction
        processor: model processor
        prompt_type: "tianxi" (default), "qwen3", or "vlmevalkit"
    """
    x, y = padded_size
    
    if prompt_type == "qwen3":
        # Use qwen3 format prompt (from testpro_qwen3.py)
        system_pr_qwen3 = get_qwen3_system_prompt(x, y)
        message = [
            {
                "role": "system",
                "content": f"{system_pr_qwen3}"
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": None},
                    {"type": "text", "text": f"{user_query}"},
                ]
            }
        ]
    elif prompt_type == "vlmevalkit":
        # Use pyautogui format for Holo2/Qwen3 models
        # Note: Holo2 outputs normalized coordinates [0-1000]
        system_pr_vlmevalkit = (
            "You are a GUI agent. You are given a task and a screenshot of the screen. "
            "You need to perform pyautogui click/moveTo action to complete the task. "
            "The answer format is `pyautogui.click(x=?, y=?), x and y is necessary`"
        )
        user_query_vlmevalkit = f"Please complete the following tasks by clicking using `pyautogui.click`:\n{user_query}"
        
        message = [
            {
                "role": "system",
                "content": system_pr_vlmevalkit
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": None},
                    {"type": "text", "text": user_query_vlmevalkit},
                ]
            }
        ]
    elif prompt_type == "holo2_json":
        # Holo2 官方 JSON 格式 (参考 联想验收/holo2_screenspot.py)
        # 输出格式: {"x": int, "y": int}，坐标范围 [0-1000]
        json_schema = {
            "properties": {
                "x": {"type": "integer", "minimum": 0, "maximum": 1000, 
                      "description": "The x coordinate, normalized between 0 and 1000."},
                "y": {"type": "integer", "minimum": 0, "maximum": 1000, 
                      "description": "The y coordinate, normalized between 0 and 1000."}
            },
            "required": ["x", "y"],
            "title": "ClickCoordinates",
            "type": "object"
        }
        
        prompt_text = (
            "Localize an element on the GUI image according to the provided target "
            "and output a click position.\n"
            f"* You must output a valid JSON following the format: {json.dumps(json_schema)}\n"
            f"Your target is:\n{user_query}"
        )
        
        # Holo2 官方格式：无 system，只有 user（图片+文本）
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": None},
                    {"type": "text", "text": prompt_text},
                ]
            }
        ]
    elif prompt_type == "qwen25vl_toolcall":
        # Qwen2.5-VL 官方 tool_call 格式 (参考 /home/zbr/wangbo/ScreenSpot-Pro-GUI-Grounding/models/qwen2_5vl.py)
        # 输出格式: <tool_call>{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [x, y]}}</tool_call>
        # 坐标为绝对像素坐标（相对于 resized 后的图像尺寸）
        
        # System prompt with tool definition
        system_content = f"""You are a helpful assistant.


# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{"type": "function", "function": {{"name_for_human": "computer_use", "name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\\n* The screen's resolution is {x}x{y}.\\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\\n* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {{"properties": {{"action": {{"description": "The action to perform. The available actions are:\\n* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\\n* `type`: Type a string of text on the keyboard.\\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\\n* `left_click`: Click the left mouse button.\\n* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\\n* `right_click`: Click the right mouse button.\\n* `middle_click`: Click the middle mouse button.\\n* `double_click`: Double-click the left mouse button.\\n* `scroll`: Performs a scroll of the mouse scroll wheel.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"], "type": "string"}}, "keys": {{"description": "Required only by `action=key`.", "type": "array"}}, "text": {{"description": "Required only by `action=type`.", "type": "string"}}, "coordinate": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click_drag`.", "type": "array"}}, "pixels": {{"description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.", "type": "number"}}, "time": {{"description": "The seconds to wait. Required only by `action=wait`.", "type": "number"}}, "status": {{"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}}}, "required": ["action"], "type": "object"}}, "args_format": "Format the arguments as a JSON object."}}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

        message = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_content}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": None},
                    {"type": "text", "text": user_query}
                ]
            }
        ]
    else:
        # Default: tianxi format prompt
        qwen_prompts = "You are a helpful assistant."
        user_prompt = f"The image is a screenshot of a computer or mobile phone interface, with a resolution of {x}x{y}. Please provide the coordinates of the object to be operated according to the command, which is as follows: {user_query}.\n"
        user_prompt_repeat = f"\nRepeat the task again for you:\nPlease provide the coordinates of the object to be operated according to the command, which is as follows: {user_query}. You must output in the following format, and the specific format is as follows: <|box_start|>(x1,y1),(x2,y2)<|box_end|>\n"
        
        message = [
            {"role": "system", "content": qwen_prompts},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{user_prompt}"},
                    {"type": "image", "image": None},
                    {"type": "text", "text": f"{user_prompt_repeat}"},
                ]
            }
        ]

    # For Holo2/Qwen3/Qwen2.5VL models, disable thinking mode to avoid <think> tags
    if prompt_type in ["vlmevalkit", "qwen3", "holo2_json", "qwen25vl_toolcall"]:
        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True, thinking=False)
    else:
        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    
    # For qwen25vl_toolcall, add guided prefix to force JSON output
    if prompt_type == "qwen25vl_toolcall":
        guide_text = '<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": ['
        text = text + guide_text
    
    return text

def process_image(img, max_size=7494400):
    """scale image and padding to multiples of 28"""
    width, height = img.size
    original_pixels = width * height
    scale_factor = 1.0

    if original_pixels > max_size:
        scale_factor = (max_size / original_pixels) ** 0.5
        new_width = max(1, int(width * scale_factor))
        new_height = max(1, int(height * scale_factor))
        logging.info(f"scale image from {width}×{height} to {new_width}×{new_height}")
    else:
        new_width, new_height = width, height

    padded_width = ((new_width // 28) + 1) * 28 if new_width % 28 != 0 else new_width
    padded_height = ((new_height // 28) + 1) * 28 if new_height % 28 != 0 else new_height

    img = img.resize((new_width, new_height), Image.LANCZOS)
    img_rgb = Image.new('RGB', (padded_width, padded_height), (0, 0, 0))
    img_rgb.paste(img, (0, 0))

    return img_rgb, scale_factor, (padded_width, padded_height)

def extract_output(output_text, prompt_type="tianxi"):
    """
    Extract output based on prompt type
    
    For tianxi format:
        skip_special_tokens=False, output: <|box_start|>(593,264),(681,354)<|box_end|><|im_end|>
        skip_special_tokens=True, output: (593,264),(681,354)
    
    For qwen3 format:
        output: <tool_call>
        {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [x, y]}}
        </tool_call>
    
    For vlmevalkit format:
        Holo2 outputs: pyautogui.click(x=?, y=?) with normalized coords [0-1000]
    
    For holo2_json format (官方推荐):
        Holo2 outputs: {"x": int, "y": int} with normalized coords [0-1000]
    """
    point_in_pixel = None
    bbx_pred = None
    
    try:
        if prompt_type == "qwen3":
            # Extract qwen3 format output
            # Note: json is already imported at the top of the file
            if '<tool_call>' in output_text and '</tool_call>' in output_text:
                json_str = output_text.split('<tool_call>')[1].split('</tool_call>')[0].strip()
                action_data = json.loads(json_str)
                coordinate = action_data.get('arguments', {}).get('coordinate', [])
                if coordinate and len(coordinate) == 2:
                    x, y = coordinate
                    # qwen3 outputs normalized coordinates (0-1000), need to keep them as is
                    point_in_pixel = (float(x), float(y))
                    # For qwen3, we create a small bbox around the point (17x17 pixels)
                    half_size = 8.5
                    bbx_pred = (
                        float(x) - half_size,
                        float(y) - half_size,
                        float(x) + half_size,
                        float(y) + half_size
                    )
        elif prompt_type == "vlmevalkit":
            # Extract pyautogui.click format for Holo2/Qwen3 models
            # Supports: pyautogui.click(502, 700) OR pyautogui.click(x=541, y=780)
            # Note: Holo2 outputs normalized coordinates [0-1000]
            
            # Try comprehensive pattern matching both formats
            # Pattern matches: x=123 OR x:123 OR x"123 OR just (123, 456) after pyautogui.click
            pattern = r"(?:x[=:\"]+([\d.]+)\s*,?\s*y[=:\"]+([\d.]+)|pyautogui\.click\(([\d.]+)\s*,\s*([\d.]+)\))"
            match = re.search(pattern, output_text)
            
            if match:
                # Group 1&2 for x=/y= format, Group 3&4 for plain (x,y) format
                if match.group(1):
                    x = float(match.group(1))
                    y = float(match.group(2))
                else:
                    x = float(match.group(3))
                    y = float(match.group(4))
                
                # Keep normalized coordinates [0-1000] as-is
                point_in_pixel = (x, y)
                # Create small bbox around the point
                half_size = 8.5
                bbx_pred = (
                    x - half_size,
                    y - half_size,
                    x + half_size,
                    y + half_size
                )
            else:
                point_in_pixel = None
                bbx_pred = None
        elif prompt_type == "holo2_json":
            # Holo2 官方 JSON 格式: {"x": int, "y": int}，坐标范围 [0-1000]
            # 参考: 联想验收/holo2_screenspot.py _parse_point_from_text()
            start = output_text.rfind("{")
            end = output_text.rfind("}") + 1
            if start != -1 and end > start:
                json_str = output_text[start:end].strip()
                data = json.loads(json_str)
                x = float(data["x"])
                y = float(data["y"])
                # Clamp to valid range [0, 1000]
                x = max(0.0, min(1000.0, x))
                y = max(0.0, min(1000.0, y))
                point_in_pixel = (x, y)
                # Create small bbox around the point
                half_size = 8.5
                bbx_pred = (
                    x - half_size,
                    y - half_size,
                    x + half_size,
                    y + half_size
                )
            else:
                point_in_pixel = None
                bbx_pred = None
        elif prompt_type == "qwen25vl_toolcall":
            # Qwen2.5-VL tool_call 格式: <tool_call>{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [x, y]}}</tool_call>
            # 坐标为绝对像素坐标（相对于 resized 后的图像尺寸）
            # 参考: /home/zbr/wangbo/ScreenSpot-Pro-GUI-Grounding/models/qwen2_5vl.py
            
            # Reconstruct the full response with guided prefix
            guide_text = '<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": ['
            full_response = guide_text + output_text
            
            # Trim to valid JSON end
            cut_index = full_response.rfind('}')
            if cut_index != -1:
                full_response = full_response[:cut_index + 1]
            
            try:
                # Parse the JSON from tool_call
                if '<tool_call>' in full_response:
                    json_str = full_response.split('<tool_call>\n')[1]
                    if '\n</tool_call>' in json_str:
                        json_str = json_str.split('\n</tool_call>')[0]
                    action_data = json.loads(json_str)
                    coordinates = action_data.get('arguments', {}).get('coordinate', [])
                    
                    if len(coordinates) == 2:
                        x, y = coordinates
                        point_in_pixel = (float(x), float(y))
                        # Create small bbox around the point
                        half_size = 8.5
                        bbx_pred = (
                            float(x) - half_size,
                            float(y) - half_size,
                            float(x) + half_size,
                            float(y) + half_size
                        )
                    elif len(coordinates) == 4:
                        x1, y1, x2, y2 = coordinates
                        point_x = (x1 + x2) / 2
                        point_y = (y1 + y2) / 2
                        point_in_pixel = (point_x, point_y)
                        bbx_pred = (float(x1), float(y1), float(x2), float(y2))
                    else:
                        point_in_pixel = None
                        bbx_pred = None
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                logging.warning(f"[qwen25vl_toolcall] Failed to parse JSON: {e}, response: {full_response[:200]}")
                point_in_pixel = None
                bbx_pred = None
        elif prompt_type == "uground":
            # UGround 输出格式:
            # 1. bbox: <|box_start|>(x1,y1),(x2,y2)<|box_end|>
            # 2. point: (x, y) 或 x, y 等数字组合
            # 注意: 输出的是 resized image 的像素坐标 (经过 pre_resize_by_width 处理)
            
            # 首先尝试解析 bbox 格式
            box_pattern = r"<\|box_start\|\>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|\>"
            box_matches = re.findall(box_pattern, output_text)
            
            if box_matches:
                # 取最后一个匹配
                last_match = box_matches[-1]
                x1, y1, x2, y2 = [int(m) for m in last_match]
                bbx_pred = (x1, y1, x2, y2)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                point_in_pixel = (center_x, center_y)
                logging.info(f"[UGround] Parsed bbox: {bbx_pred}, point: {point_in_pixel}")
            else:
                # 尝试解析点坐标 (x, y)
                floats = re.findall(r'-?\d+\.?\d*', output_text)
                floats = [float(num) for num in floats]
                
                if len(floats) >= 4:
                    # 可能是 bbox 格式但没有标记
                    x1, y1, x2, y2 = floats[:4]
                    bbx_pred = (x1, y1, x2, y2)
                    point_in_pixel = ((x1 + x2) / 2, (y1 + y2) / 2)
                    logging.info(f"[UGround] Parsed as bbox (no markers): {bbx_pred}, point: {point_in_pixel}")
                elif len(floats) >= 2:
                    # 点坐标
                    x, y = floats[:2]
                    point_in_pixel = (x, y)
                    # 创建小 bbox
                    half_size = 8.5
                    bbx_pred = (x - half_size, y - half_size, x + half_size, y + half_size)
                    logging.info(f"[UGround] Parsed point: {point_in_pixel}, bbox: {bbx_pred}")
                else:
                    logging.warning(f"[UGround] Failed to parse coordinates from: {output_text[:200]}")
                    point_in_pixel = None
                    bbx_pred = None
        else:
            # Extract tianxi format output (bbox format)
            pattern = r'\((\d+),(\d+)\)'
            matches = re.findall(pattern, output_text)
            if len(matches) == 2:
                coords = [int(c) for m in matches for c in m]
                bbx_pred = tuple(coords)
                x1, y1, x2, y2 = bbx_pred
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                point_in_pixel = (center_x, center_y)
    except Exception as e:
        logging.error("wrong_format in extract_output (prompt_type=%s): %s", prompt_type, e)

    return point_in_pixel, bbx_pred

def compute_ground_result(img, text, model, processor, prompt_type="tianxi",
                          do_sample=False, temperature=1.0, top_p=0.9, seed=None):
    """
    执行 grounding 模型前向传播
    
    新增 sampling 参数：
    - do_sample: 是否启用采样 (True=sampling, False=greedy)
    - temperature: 采样温度 (默认 1.0)
    - top_p: nucleus sampling 阈值 (默认 0.9)
    - seed: 随机种子 (用于复现性)
    """
    # model forward
    inputs = processor(
        text=[text], images=[img], 
        max_length=40000, truncation=False, 
        padding=True, return_tensors="pt")
    
    # 支持多卡模型（device_map="auto"）：移动 inputs 到模型的第一个设备
    if hasattr(model, 'device'):
        inputs = inputs.to(model.device)
    else:
        # 多卡模型没有单一 device，使用 hf_device_map 获取第一个设备
        device = next(model.parameters()).device
        inputs = inputs.to(device)
    
    # Increase max_new_tokens for Qwen3 CoT models (they use <think> tags)
    # holo2_json 和 qwen25vl_toolcall 格式只需要 64 tokens（guided decoding），其他格式需要更多
    max_tokens = 64 if prompt_type in ("holo2_json", "qwen25vl_toolcall") else 500
    output_dict = generate(
        model=model, **inputs, max_new_tokens=max_tokens, return_scores=True,
        do_sample=do_sample, temperature=temperature, top_p=top_p, seed=seed
    )
    # extract results
    output_ids = output_dict["output_ids"]
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids 
        in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)[0]
    point_pred, bbox_pred = extract_output(output_text, prompt_type=prompt_type)

    return dict(
        point_pred=point_pred,
        bbox_pred=bbox_pred,
        output_text=output_text
    )


def compute_ground_result_uground(img, user_query, uground_model, 
                                   do_sample=False, temperature=1.0, top_p=0.9, seed=None):
    """
    UGround 模型专用推理函数
    
    UGround 使用 LLaVA 架构，接口与 Qwen 完全不同
    
    Args:
        img: PIL Image (原始图像)
        user_query: 用户指令
        uground_model: UGroundModel 实例
        do_sample: 是否采样
        temperature: 采样温度
        top_p: nucleus sampling
        seed: 随机种子
    
    Returns:
        dict with point_pred (pixel coords), bbox_pred (pixel coords), output_text
    """
    import torch
    from .models.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from .models.llava.conversation import conv_templates
    from .models.llava.mm_utils import tokenizer_image_token, process_images, pre_resize_by_width
    
    # 保存原始图像尺寸
    orig_width, orig_height = img.size
    
    # 构建 prompt
    prompt_template = "In the screenshot, where are the pixel coordinates (x, y) of the element corresponding to \"{}\"?"
    full_prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt_template.format(user_query)
    conv = conv_templates['llava_v1'].copy()
    conv.append_message(conv.roles[0], full_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # 处理输入
    input_ids = tokenizer_image_token(
        prompt, uground_model.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).cuda()
    
    # Resize image and prepare tensor for inference
    resized_image, pre_resize_scale = pre_resize_by_width(img)
    image_tensor, image_new_size = process_images(
        [resized_image], uground_model.image_processor, uground_model.model.config
    )
    
    # 推理
    with torch.inference_mode():
        output_ids = uground_model.model.generate(
            input_ids,
            images=image_tensor.half().cuda(),
            image_sizes=[image_new_size],
            do_sample=do_sample,
            temperature=temperature if do_sample else 0,
            top_p=top_p,
            num_beams=1,
            max_new_tokens=128,
            use_cache=True
        )
    
    response = uground_model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    logging.info(f"[UGround] Raw response: {response}")
    
    # 解析输出 - UGround 输出的是 resized image 的像素坐标
    point_pred, bbox_pred = extract_output(response, prompt_type='uground')
    
    # 坐标转换: 从 resized image 坐标 -> 原始图像坐标
    if point_pred is not None:
        # 先除以 pre_resize_scale 得到原始像素坐标
        point_pred = (
            point_pred[0] / pre_resize_scale,
            point_pred[1] / pre_resize_scale
        )
        logging.info(f"[UGround] Point after scale conversion: {point_pred}")
    
    if bbox_pred is not None:
        # 先除以 pre_resize_scale 得到原始像素坐标
        bbox_pred = tuple(coord / pre_resize_scale for coord in bbox_pred)
        logging.info(f"[UGround] BBox after scale conversion: {bbox_pred}")
    
    return dict(
        point_pred=point_pred,
        bbox_pred=bbox_pred,
        output_text=response,
        pre_resize_scale=pre_resize_scale,  # 保存缩放比例供调试
        orig_size=(orig_width, orig_height)
    )


class BaseAction:
    _required_keys_ = ["image"]
    
    def check_output(self, output_dict):
        for key in self._required_keys_:
            if key not in output_dict:
                raise ValueError(f"{key} not found in output_dict")

    def __call__(self, input_dict, model_dict):
        output_dict = self.compute(input_dict, model_dict)
        self.check_output(output_dict)
        return output_dict

    def compute(self, input_dict, model_dict) -> Dict:
        raise NotImplementedError

class Grounding(BaseAction):
    _required_keys_ = ["image", "bbox_abs"]

    def __init__(self, prompt_type="tianxi", model_type="qwen"):
        """
        Initialize Grounding action
        
        Args:
            prompt_type: "tianxi" (default), "qwen3", "vlmevalkit", "holo2_json", "qwen25vl_toolcall", "uground"
            model_type: "qwen" (default) or "uground"
        """
        super().__init__()
        self.prompt_type = prompt_type
        self.model_type = model_type

    def compute(self, input_dict, model_dict):
        # prepare input
        assert len(input_dict) == 1, f"The length of input_dict should be 1, but got {len(input_dict)}"
        input_dict = input_dict[0]
        image = input_dict["image"]
        user_query = input_dict["user_query"]
        coord_abs = input_dict["coord_abs"]
        model = model_dict["model"]
        processor = model_dict["processor"]

        # compute
        output_dict = deepcopy(input_dict)
        
        if self.model_type == "uground":
            # UGround 模型: 使用专用推理函数
            # UGround 内部处理图像缩放，返回的坐标已经是原始图像像素坐标
            ground_result = compute_ground_result_uground(image, user_query, model)
            output_dict.update(**ground_result)
            
            # UGround 返回的坐标已经是原始图像像素坐标，不需要 scale 转换
            if coord_abs is None:
                a_x1, a_y1 = 0, 0
            else:
                assert len(coord_abs) == 4
                a_x1, a_y1, *_ = coord_abs
            
            if output_dict["bbox_pred"] is not None:
                r_x1, r_y1, r_x2, r_y2 = output_dict["bbox_pred"]
                # UGround 坐标已经是原始像素坐标，直接加上偏移
                x1 = a_x1 + r_x1
                y1 = a_y1 + r_y1
                x2 = a_x1 + r_x2
                y2 = a_y1 + r_y2
                output_dict["bbox_abs"] = (x1, y1, x2, y2)
                output_dict["point_abs"] = ((x1 + x2) / 2, (y1 + y2) / 2)
            else:
                output_dict["bbox_abs"] = None
                output_dict["point_abs"] = None
            
            # 对于 UGround，image 保持原样，rescale 设为 1.0
            output_dict["image"] = image
            output_dict["rescale"] = 1.0
        else:
            # Qwen 系列模型: 使用原有流程
            img, scale, padded_size = process_image(image)
            text = get_input(padded_size, user_query, processor, prompt_type=self.prompt_type)
            ground_result = compute_ground_result(img, text, model, processor, prompt_type=self.prompt_type)
            output_dict.update(**ground_result)

            if coord_abs is None:
                a_x1, a_y1 = 0, 0
            else:
                assert len(coord_abs) == 4 # (x1, y1, x2, y2)
                a_x1, a_y1, *_ = coord_abs
            
            if output_dict["bbox_pred"] is not None:
                r_x1, r_y1, r_x2, r_y2 = output_dict["bbox_pred"]
                
                # 🔧 For vlmevalkit/holo2_json prompts (Holo2 models): coordinates are normalized [0-1000]
                # Need to convert to actual pixels: (coord / 1000) * image_dimension
                if self.prompt_type in ("vlmevalkit", "holo2_json"):
                    # Get padded image dimensions
                    img_width, img_height = img.size
                    # Convert from [0-1000] to actual pixels
                    r_x1 = (r_x1 / 1000.0) * img_width
                    r_y1 = (r_y1 / 1000.0) * img_height
                    r_x2 = (r_x2 / 1000.0) * img_width
                    r_y2 = (r_y2 / 1000.0) * img_height
                    logging.info(f"[DEBUG] Converted normalized coords [{self.prompt_type}] to pixels: ({r_x1:.1f}, {r_y1:.1f}, {r_x2:.1f}, {r_y2:.1f})")
                
                # Restore to original image scale
                r_x1 = r_x1 / scale
                r_y1 = r_y1 / scale
                r_x2 = r_x2 / scale
                r_y2 = r_y2 / scale
                
                # Add absolute offset
                x1 = a_x1 + r_x1
                y1 = a_y1 + r_y1
                x2 = a_x1 + r_x2
                y2 = a_y1 + r_y2
                output_dict["bbox_abs"] = (x1, y1, x2, y2)
                output_dict["point_abs"] = ((x1 + x2) / 2, (y1 + y2) / 2)
                output_dict["bbox_pred"] = (r_x1, r_y1, r_x2, r_y2)
            else:
                output_dict["bbox_abs"] = None
                output_dict["point_abs"] = None
            output_dict["image"] = img
            output_dict["rescale"] = scale
        
        return output_dict

class MaskGrounding(Grounding):
    def compute(self, input_dict, model_dict):
        output_dict = super(MaskGrounding, self).compute(input_dict, model_dict)
        # mask image
        if output_dict["bbox_pred"] is not None:
            mask_bbox = output_dict["bbox_pred"]
            image = output_dict["image"]
            output_dict["image"] = mask_image(image, mask_bbox)
        return output_dict


class RandomSamplingGrounding(BaseAction):
    """
    Random Sampling Grounding: 使用温度采样生成候选 bbox
    用于对照实验：证明 Sequential Masking 比 Random Sampling 更有效
    """
    _required_keys_ = ["image", "bbox_abs"]

    def __init__(self, prompt_type="tianxi", temperature=1.0, top_p=0.9, seed=42):
        """
        Initialize RandomSamplingGrounding action
        
        Args:
            prompt_type: "tianxi" (default) or "qwen3" etc.
            temperature: 采样温度 (>1.0 增加随机性, <1.0 减少随机性)
            top_p: nucleus sampling 阈值
            seed: 随机种子 (保证可复现性)
        """
        super().__init__()
        self.prompt_type = prompt_type
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed

    def compute(self, input_dict, model_dict):
        # prepare input
        assert len(input_dict) == 1, f"The length of input_dict should be 1, but got {len(input_dict)}"
        input_dict = input_dict[0]
        image = input_dict["image"]
        user_query = input_dict["user_query"]
        coord_abs = input_dict["coord_abs"]
        model = model_dict["model"]
        processor = model_dict["processor"]

        # compute with sampling
        output_dict = deepcopy(input_dict)
        img, scale, padded_size = process_image(image)
        text = get_input(padded_size, user_query, processor, prompt_type=self.prompt_type)
        
        # 🔧 使用 sampling 而非 greedy
        ground_result = compute_ground_result(
            img, text, model, processor, prompt_type=self.prompt_type,
            do_sample=True, temperature=self.temperature, top_p=self.top_p, seed=self.seed
        )
        output_dict.update(**ground_result)
        
        logging.info(f"[RandomSamplingGrounding] temp={self.temperature}, top_p={self.top_p}, seed={self.seed}")

        if coord_abs is None:
            a_x1, a_y1 = 0, 0
        else:
            assert len(coord_abs) == 4 # (x1, y1, x2, y2)
            a_x1, a_y1, *_ = coord_abs
        
        if output_dict["bbox_pred"] is not None:
            r_x1, r_y1, r_x2, r_y2 = output_dict["bbox_pred"]
            
            # 🔧 For vlmevalkit/holo2_json prompts (Holo2 models): coordinates are normalized [0-1000]
            # Need to convert to actual pixels: (coord / 1000) * image_dimension
            if self.prompt_type in ("vlmevalkit", "holo2_json"):
                # Get padded image dimensions
                img_width, img_height = img.size
                # Convert from [0-1000] to actual pixels
                r_x1 = (r_x1 / 1000.0) * img_width
                r_y1 = (r_y1 / 1000.0) * img_height
                r_x2 = (r_x2 / 1000.0) * img_width
                r_y2 = (r_y2 / 1000.0) * img_height
                logging.info(f"[DEBUG] Converted normalized coords [{self.prompt_type}] to pixels: ({r_x1:.1f}, {r_y1:.1f}, {r_x2:.1f}, {r_y2:.1f})")
            
            # Restore to original image scale
            r_x1 = r_x1 / scale
            r_y1 = r_y1 / scale
            r_x2 = r_x2 / scale
            r_y2 = r_y2 / scale
            
            # Add absolute offset
            x1 = a_x1 + r_x1
            y1 = a_y1 + r_y1
            x2 = a_x1 + r_x2
            y2 = a_y1 + r_y2
            output_dict["bbox_abs"] = (x1, y1, x2, y2)
            output_dict["point_abs"] = ((x1 + x2) / 2, (y1 + y2) / 2)
            output_dict["bbox_pred"] = (r_x1, r_y1, r_x2, r_y2)
        else:
            output_dict["bbox_abs"] = None
            output_dict["point_abs"] = None
        output_dict["image"] = img
        output_dict["rescale"] = scale
        
        return output_dict

class Crop(BaseAction):
    _required_keys_ = ["image"]

    def __init__(self, crop_ratio: float = 0.25):
        super(Crop, self).__init__()
        self.crop_ratio = crop_ratio

    def compute(self, input_dict, model_dict):
        # prepare input
        assert len(input_dict) > 1
        output_dict = deepcopy(input_dict[0])
        base_image = output_dict["image"]
        ref_bbox = []
        for idx in range(1, len(input_dict)):
            cur_bbox_pred = input_dict[idx]["bbox_pred"]
            if cur_bbox_pred is not None:
                ref_bbox.append(input_dict[idx]["bbox_pred"])
        # crop image
        bbox_crop = compute_crop_box(ref_bbox, base_image.size, ratio=self.crop_ratio)
        
        # Additional safety check
        x1, y1, x2, y2 = bbox_crop
        if x1 >= x2 or y1 >= y2:
            logging.warning(f"Invalid crop box: {bbox_crop}, using full image")
            bbox_crop = (0, 0, base_image.width, base_image.height)
        
        crop_image = base_image.crop(bbox_crop)
        output_dict["image"] = crop_image
        output_dict["bbox_crop"] = bbox_crop
        output_dict["coord_abs"] = bbox_crop

        return output_dict


class DrawDualBoxesSeparate(BaseAction):
    """Draw two bounding boxes separately and save as two independent images, each centered on bbox with 20% expansion"""
    _required_keys_ = ["image"]
    
    def compute(self, input_dict, model_dict):
        # Get bbox from two inputs
        assert len(input_dict) == 2, "DrawDualBoxesSeparate requires two inputs"
        
        bbox1 = input_dict[0].get("bbox_abs")
        bbox2 = input_dict[1].get("bbox_abs")
        # Get original image from pipeline initial state
        base_image = model_dict.get("original_image", input_dict[0]["image"])
        user_query = input_dict[0]["user_query"]
        
        logging.info(f"DrawDualBoxesSeparate - Processing two bounding boxes")
        logging.info(f"Box 1 (First grounding): {bbox1}")
        logging.info(f"Box 2 (After mask regrounding): {bbox2}")
        
        # Validate bbox
        if bbox1 is not None and len(bbox1) == 4:
            if bbox1[0] >= bbox1[2] or bbox1[1] >= bbox1[3]:
                logging.warning(f"Box 1 coordinates invalid: {bbox1}")
                bbox1 = None
        
        if bbox2 is not None and len(bbox2) == 4:
            if bbox2[0] >= bbox2[2] or bbox2[1] >= bbox2[3]:
                logging.warning(f"Box 2 coordinates invalid: {bbox2}")
                bbox2 = None
        
        from PIL import ImageDraw, ImageFont, Image
        
        # Get original image dimensions
        img_width, img_height = base_image.size
        
        # Calculate expansion dimensions (20% of original image size)
        expand_width = int(img_width * 0.2)
        expand_height = int(img_height * 0.2)
        logging.info(f"Image expansion size: {expand_width}x{expand_height} (20% of original)")
        
        # Process first bbox
        img1 = self._process_single_bbox(base_image, bbox1, "1", (0, 255, 0), expand_width, expand_height)
        
        # Process second bbox
        img2 = self._process_single_bbox(base_image, bbox2, "2", (255, 0, 0), expand_width, expand_height)
        
        # Return single output dict containing two images
        output_dict = deepcopy(input_dict[0])
        output_dict["image1"] = img1
        output_dict["image2"] = img2
        output_dict["bbox1"] = bbox1
        output_dict["bbox2"] = bbox2
        output_dict["user_query"] = user_query
        # Keep image field to satisfy BaseAction requirements
        output_dict["image"] = img1  # Default to first image
        
        return output_dict
    
    def _process_single_bbox(self, base_image, bbox, label, color, expand_width, expand_height):
        """Process single bbox, draw and expand around center"""
        from PIL import ImageDraw, ImageFont, Image
        
        if bbox is None:
            # If no bbox, return original image
            return base_image.copy()
        
        # Copy image
        img = base_image.copy()
        
        # Create transparent layer
        overlay = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        
        # Try to load font
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Draw bounding box and label
        draw_overlay.rectangle(bbox, outline=color + (255,), width=3)
        draw_overlay.text((bbox[0], bbox[1] - 30), label, fill=color + (255,), font=font)
        
        # Blend layers
        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        
        # Calculate bbox center
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Calculate crop region (centered on bbox with 20% expansion)
        crop_left = max(0, int(center_x - expand_width))
        crop_top = max(0, int(center_y - expand_height))
        crop_right = min(img.width, int(center_x + expand_width))
        crop_bottom = min(img.height, int(center_y + expand_height))
        
        # Ensure crop region is valid
        if crop_right <= crop_left:
            # If width invalid, ensure minimum width
            if center_x < img.width / 2:
                crop_right = min(img.width, crop_left + 100)  # At least 100 pixels wide
            else:
                crop_left = max(0, crop_right - 100)
        
        if crop_bottom <= crop_top:
            # If height invalid, ensure minimum height
            if center_y < img.height / 2:
                crop_bottom = min(img.height, crop_top + 100)  # At least 100 pixels high
            else:
                crop_top = max(0, crop_bottom - 100)
        
        # Crop image
        cropped_img = img.crop((crop_left, crop_top, crop_right, crop_bottom))
        
        return cropped_img




class GPTJudgeTwoImages(BaseAction):
    """Use GPT/OpenRouter to judge which image selection is better"""
    _required_keys_ = ["image", "image1", "image2"]
    
    def __init__(self, api_key=None, base_url=None, model="openai/gpt-4o", site_url=None, site_title=None):
        super().__init__()
        # Base URL: OpenRouter uses https://openrouter.ai/api/v1; OpenAI uses https://api.openai.com/v1
        self.base_url = base_url or os.environ.get("OPENROUTER_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://openrouter.ai/api/v1"
        # API Key: OpenRouter prioritizes OPENROUTER_API_KEY
        if "openrouter.ai" in self.base_url:
            self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
            # OpenRouter model names usually follow provider/model format, e.g. openai/gpt-4o
            if model and "/" not in model:
                model = f"openai/{model}"
        else:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.site_url = site_url or os.environ.get("OPENROUTER_SITE_URL")
        self.site_title = site_title or os.environ.get("OPENROUTER_SITE_TITLE") or "GUI-Agent"
    
    def compute(self, input_dict, model_dict):
        # Process single input (from DrawDualBoxesSeparate)
        assert len(input_dict) == 1
        input_dict = input_dict[0]
        
        # Get two images
        image1 = input_dict["image1"]
        image2 = input_dict["image2"] 
        bbox1 = input_dict.get("bbox1")
        bbox2 = input_dict.get("bbox2")
        user_query = input_dict["user_query"]
        
        logging.info(f"GPTJudgeTwoImages - Starting to judge two images")
        logging.info(f"User query: {user_query}")
        logging.info(f"Box 1: {bbox1}")
        logging.info(f"Box 2: {bbox2}")
        
        # Call GPT API to compare two images
        selected_image, reason, response_text = self.judge_two_images(
            image1, image2, user_query
        )
        
        logging.info(f"GPT judgment result: Selected image {selected_image}")
        logging.info(f"GPT judgment reason:\n{reason}")
        
        # Build output
        output_dict = deepcopy(input_dict)
        output_dict["selected_image"] = selected_image
        output_dict["judge_reason"] = reason
        output_dict["judge_response"] = response_text
        
        # Set final bbox and point
        if selected_image == "1" and bbox1:
            output_dict["bbox_abs"] = bbox1
            output_dict["point_abs"] = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
            output_dict["image"] = image1  # Set selected image
        elif selected_image == "2" and bbox2:
            output_dict["bbox_abs"] = bbox2
            output_dict["point_abs"] = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
            output_dict["image"] = image2  # Set selected image
        else:
            # If no valid selection, keep bbox1
            output_dict["bbox_abs"] = bbox1
            output_dict["point_abs"] = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2) if bbox1 else None
            output_dict["image"] = image1
        
        return output_dict
    
    def judge_two_images(self, image1, image2, user_query):
        """Use GPT to judge which image better meets user requirements"""
        try:
            from openai import OpenAI
        except ImportError:
            logging.error("Please install openai library: pip install openai")
            raise
        
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        prompt = f"""You are comparing two images to determine which one better fulfills the user's intent.

User Command: "{user_query}"

Image 1: Shows a GUI element marked with a green box labeled "1"
Image 2: Shows a GUI element marked with a red box labeled "2"

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
- If one shows a selected state and the other shows normal state, prefer the normal state
- ELEMENT QUALITY HIERARCHY (best to worst):
   - Icon + Text together (most informative and complete)
   - Complete icon alone (clear visual indicator)  
   - Complete text alone (readable label)
   - Multiple elements in one box OR incomplete elements (ambiguous target)

COMMON PITFALLS TO AVOID:
    - Don't choose based on keyword matching alone
    - Don't overlook the user's actual goal in favor of literal interpretation

Remember: Provide SPECIFIC analysis based on what you actually observe, not generic descriptions.

**OUTPUT FORMAT**:
<analysis>
Image 1: [Describe what element is highlighted and its purpose]
Image 2: [Describe what element is highlighted and its purpose]
Comparison: [Explain which better serves the user's intent and why]
</analysis>

<answer>1 or 2</answer>
<reason>Brief explanation of why this image shows the better choice</reason>"""

        try:
            import base64
            from io import BytesIO
            
            # Convert images to base64
            buffered1 = BytesIO()
            image1.save(buffered1, format="PNG")
            img1_base64 = base64.b64encode(buffered1.getvalue()).decode()
            
            buffered2 = BytesIO()
            image2.save(buffered2, format="PNG")
            img2_base64 = base64.b64encode(buffered2.getvalue()).decode()
            
            extra_headers = None
            if "openrouter.ai" in self.base_url:
                extra_headers = {
                    "HTTP-Referer": self.site_url or "https://localhost",
                    "X-Title": self.site_title,
                }
            
            response = client.chat.completions.create(
                model=self.model,
                extra_headers=extra_headers,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img1_base64}",
                                    "detail": "high"
                                }
                            },
                            {
                                "type": "image_url", 
                                "image_url": {
                                    "url": f"data:image/png;base64,{img2_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                temperature=0,
                max_tokens=9600
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            import re
            
            # Extract analysis and answer
            analysis_match = re.search(r'<analysis>(.*?)</analysis>', response_text, re.DOTALL)
            answer_match = re.search(r'<answer>(\d)</answer>', response_text)
            reason_match = re.search(r'<reason>(.*?)</reason>', response_text, re.DOTALL)
            
            selected_image = answer_match.group(1) if answer_match else "1"
            reason = reason_match.group(1).strip() if reason_match else "No reason provided"
            
            # If there's analysis content, include it in the reason
            if analysis_match:
                analysis = analysis_match.group(1).strip()
                logging.info(f"GPT detailed analysis:\n{analysis}")
                reason = f"{analysis}\n\nFinal selection: {reason}"
            
        except Exception as e:
            logging.error(f"GPT API call failed: {e}")
            selected_image = "1"
            reason = f"API call failed: {str(e)}"
            response_text = str(e)
        
        return selected_image, reason, response_text

