import copy
import itertools
import time

import torch
import json
import re
import argparse
import os
from PIL import Image
import logging
from tqdm import tqdm

from model_factory import build_model

logging.basicConfig(level=logging.INFO)
torch.manual_seed(114514)

GT_TYPES = ['positive', 'negative']
INSTRUCTION_STYLES = ['instruction', 'action', 'description']
LANGUAGES = ['en', 'cn']
GAZE_MODES = ['none', 'raw', 'norm']
TASK_TYPES = ['Simple Grounding', 'Semantic-matching', 'Spatial-anchoring']  

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--model_name_or_path', type=str, required=False)
    parser.add_argument('--screenspot_imgs', type=str, required=True)
    parser.add_argument('--screenspot_test', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--inst_style', type=str, required=True, choices=INSTRUCTION_STYLES + ['all'], help="Instruction style to use.")
    parser.add_argument('--language', type=str, required=True, choices=LANGUAGES + ['all'], default='en', help="Language to use.")
    parser.add_argument('--gt_type', type=str, required=True, choices=GT_TYPES + ['all'], help="Ground truth type: 'positive' or 'negative'.")
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--max_pixels', type=int, default=3840 * 2160, help="Maximum number of pixels for the model to process.")
    parser.add_argument(
        '--gaze_mode', type=str, default='none', choices=GAZE_MODES,
        help=(
            "Gaze point injection mode: "
            "'none' = no injection; "
            "'raw' = prepend 'I look at [x, y] in this image.' using original pixel coords; "
            "'norm' = prepend 'I look at [x*1000/W, y*1000/H] in this image.' using normalized coords (0~1000)."
        )
    )
    
    parser.add_argument(
        '--task_type', type=str, default='all',
        help=(
            "按 task_type 过滤测试样本。"
            f"可选值: {TASK_TYPES} 或逗号分隔多个，'all' 表示不过滤。"
            "示例: --task_type 'Semantic-matching' "
            "或 --task_type 'Simple Grounding,Spatial-anchoring'"
        )
    )

    args = parser.parse_args()
    return args


def collect_results_to_eval(results, platform=None, scenario=None, place=None, language=None, gt_type=None, instruction_style=None, ui_type=None, activity=None, is_same_window=None, task_type=None):
    """
    Filters the results based on provided values. None means include all (ignore filtering this attribute).

    Parameters:
        results (list): A list of dictionaries containing sample results.
    
    Returns:
        list: A filtered list of dictionaries based on the given criteria.
    """
    filtered_results = []

    for sample in results:
        # Check each filter condition; if None, consider it as passed
        if (platform is None or sample.get("platform") == platform) and \
           (scenario is None or sample.get("scenario") == scenario) and \
           (place is None or sample.get("place") == place) and \
           (language is None or sample.get("language") == language) and \
           (gt_type is None or sample.get("gt_type") == gt_type) and \
           (instruction_style is None or sample.get("instruction_style") == instruction_style) and \
           (ui_type is None or sample.get("ui_type") == ui_type) and \
           (activity is None or sample.get("activity") == activity) and \
           (is_same_window is None or sample.get("is_same_window") == is_same_window) and \
           (task_type is None or sample.get("task_type") == task_type):
            filtered_results.append(sample)

    return filtered_results


def make_combinations(results, platform=False, scenario=None, place=False, language=False, gt_type=False, instruction_style=False, ui_type=False, activity=False, is_same_window=False, task_type=False):
    """
    Returns a list of combinations of values for attributes where the corresponding parameter is set to True.
    """
    # Initialize a dictionary to store unique values for each attribute
    unique_values = {
        "platform": set(),
        "scenario": set(),
        "place": set(),
        "language": set(),
        "gt_type": set(),
        "instruction_style": set(),
        "ui_type": set(),
        "activity": set(),
        "is_same_window": set(),
        "task_type": set(),
    }

    # Collect unique values from the results
    for sample in results:
        if platform:
            unique_values["platform"].add(sample.get("platform"))
        if scenario:
            unique_values["scenario"].add(sample.get("scenario"))
        if place:
            unique_values["place"].add(sample.get("place"))
        if language:
            unique_values["language"].add(sample.get("language"))
        if gt_type:
            unique_values["gt_type"].add(sample.get("gt_type"))
        if instruction_style:
            unique_values["instruction_style"].add(sample.get("instruction_style"))
        if ui_type:
            unique_values["ui_type"].add(sample.get("ui_type"))
        if activity:
            unique_values["activity"].add(sample.get("activity"))
        if is_same_window:
            unique_values["is_same_window"].add(sample.get("is_same_window"))
        if task_type:
            unique_values["task_type"].add(sample.get("task_type"))

    # Filter out the attributes that are set to False (no need for combinations)
    filtered_values = {key: list(value) for key, value in unique_values.items() if value}
    if not filtered_values:
        return []

    # Generate all combinations of the selected attributes using itertools.product
    attribute_combinations = list(itertools.product(*filtered_values.values()))

    # Convert combinations into dictionaries with corresponding attribute names
    combinations = []
    for combination in attribute_combinations:
        combinations.append(dict(zip(filtered_values.keys(), combination)))

    return combinations


def calc_metric_for_result_list(results):
    """Calculates the metrics for a simple result list."""
    num_total = len(results)
    correct_num = sum(1 for res in results if res["correctness"] == "correct")
    wrong_format_num = sum(1 for res in results if res["correctness"] == "wrong_format")

    # Calculate text and icon specific metrics using collect_results_to_eval
    text_results = collect_results_to_eval(results, ui_type="text")
    icon_results = collect_results_to_eval(results, ui_type="icon")

    text_correct = sum(1 for res in text_results if res["correctness"] == "correct")
    text_total = len(text_results)
    icon_correct = sum(1 for res in icon_results if res["correctness"] == "correct")
    icon_total = len(icon_results)
    metrics = {
        "num_correct_action": correct_num,
        "num_total": num_total,
        "wrong_format_num": wrong_format_num,
        "action_acc": correct_num / num_total if num_total > 0 else 0,
        "text_acc": text_correct / text_total if text_total > 0 else 0,
        "icon_acc": icon_correct / icon_total if icon_total > 0 else 0
    }
    return metrics


def eval_sample_positive_gt(sample, response):
    bbox = sample["bbox"]
    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]  # x1, y1, x2, y2
    # bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # x1, y1, w, h
    img_size = sample["img_size"]
    bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]
    
    click_point = response["point"]  # may be none
    print(click_point)
    if click_point is None:
        return "wrong_format"
    # Check if the predicted point falls in the ground truth box
    if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
        return "correct"
    else:
        return "wrong"
    
def eval_sample_negative_gt(sample, response):
    if response["result"] == "negative":
        return "correct"
    elif response["result"] == "positive":
        return "wrong"
    else: ## response["result"] == wrong_format
        return "wrong_format"

def evaluate_fine_grained(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        platform=True, 
        place=True,
        instruction_style=True, 
        gt_type=True
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        platform = combo.get("platform")
        place = combo.get("place")
        inst_style = combo.get("instruction_style")
        gt_type = combo.get("gt_type")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            platform=platform,
            place=place,
            instruction_style=inst_style,
            gt_type=gt_type
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"plat:{platform} place:{place} inst_style:{inst_style} gt_type:{gt_type}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_seeclick_paper_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        platform=True, 
        instruction_style=True, 
        gt_type=True
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        platform = combo.get("platform")
        inst_style = combo.get("instruction_style")
        gt_type = combo.get("gt_type")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            platform=platform,
            instruction_style=inst_style,
            gt_type=gt_type
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"plat:{platform} inst_style:{inst_style} gt_type:{gt_type}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_leaderboard_detailed_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        place=True,
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        place = combo.get("place")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            place=place,
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"place:{place}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_leaderboard_simple_style(results):
    # Generate all combinations of platform, instruction_style, and gt_type
    combinations = make_combinations(
        results, 
        scenario=True,
    )

    evaluation_result = {}

    # Iterate through each combination
    for combo in combinations:
        scenario = combo.get("scenario")
        
        # Filter results for the current combination
        filtered_results = collect_results_to_eval(
            results=results,
            scenario=scenario,
        )
        
        # Calculate metrics using the calc_metric_for_result_list function
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        
        # Construct a unique key based on the combination
        key = f"scenario:{scenario}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_task_type(results):
    """
    按 task_type 统计准确率
    """
    combinations = make_combinations(
        results,
        task_type=True,
    )

    evaluation_result = {}
    for combo in combinations:
        task_type = combo.get("task_type")
        filtered_results = collect_results_to_eval(
            results=results,
            task_type=task_type,
        )
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics["num_total"] == 0:
            continue

        key = f"task_type:{task_type}"
        evaluation_result[key] = metrics

    return evaluation_result

def evaluate_overall(results):
    """
    Evaluates the overall metrics for all results without any filtering.
    
    Parameters:
        results (list): A list of dictionaries containing sample results.
        
    Returns:
        dict: A dictionary containing the overall metrics.
    """
    # Calculate metrics for the entire result set
    metrics = calc_metric_for_result_list(results)
    
    return metrics


def evaluate(results):
    """Collect results and calculate metrics. You can comment out function calls or add new ones based on your need.
    """
    result_report = {
        "details": [],  # Store detailed information for each sample
        "metrics": {}
    }

    # TODO: comment out function calls based on your need
    result_report["metrics"]["fine_grained"] = evaluate_fine_grained(results)
    result_report["metrics"]["seeclick_style"] = evaluate_seeclick_paper_style(results)
    result_report["metrics"]["leaderboard_simple_style"] = evaluate_leaderboard_simple_style(results)
    result_report["metrics"]["leaderboard_detailed_style"] = evaluate_leaderboard_detailed_style(results)
    result_report["metrics"]["task_type"] = evaluate_task_type(results)  
    result_report["metrics"]["overall"] = evaluate_overall(results)

    # Save detailed results
    result_report["details"] = results

    return result_report

def _convert_percent_bbox_to_xyxy(target_bbox):
    """
    target_bbox: percent-based bbox with keys x,y,width,height (0~100) and original_width/height
    return: [x1,y1,x2,y2] in pixels
    """
    if target_bbox is None:
        return None
    ow = target_bbox.get("original_width")
    oh = target_bbox.get("original_height")
    if ow is None or oh is None:
        return None

    x = target_bbox.get("x")
    y = target_bbox.get("y")
    w = target_bbox.get("width")
    h = target_bbox.get("height")
    if x is None or y is None or w is None or h is None:
        return None

    x1 = x / 100.0 * ow
    y1 = y / 100.0 * oh
    x2 = (x + w) / 100.0 * ow
    y2 = (y + h) / 100.0 * oh
    return [x1, y1, x2, y2]


def _normalize_new_annotation(task_instance):
    """
    Convert new annotation format to the expected internal format.
    """
    target_bbox = task_instance.get("target_bbox")
    bbox = _convert_percent_bbox_to_xyxy(target_bbox)
    img_size = None
    if target_bbox and target_bbox.get("original_width") and target_bbox.get("original_height"):
        img_size = [target_bbox["original_width"], target_bbox["original_height"]]

    instruction_en = task_instance.get("instruction_en")
    instruction_cn = task_instance.get("instruction_cn")
    choices = task_instance.get("choices") or {}

    return {
        "id": task_instance.get("annotation_id"),
        "img_filename": task_instance.get("image"),
        "instruction": instruction_en,
        "instruction_cn": instruction_cn,
        "img_size": img_size,
        "bbox": bbox,
        "gaze_point": task_instance.get("gaze_point"),  
        "ui_type": choices.get("ui_type"),
        "platform": choices.get("platform"),
        "place": choices.get("place"),
        "scenario": choices.get("scenario"),
        "activity": choices.get("activity"),
        "is_same_window": choices.get("is_same_window"),
        "task_type": choices.get("task type"),
    }


def build_prompt_with_gaze(instruction, gaze_point, img_size, gaze_mode):
    """
    根据 gaze_mode 在 instruction 前注入 gaze 坐标前缀。

    Args:
        instruction (str): 原始指令文本。
        gaze_point (list|None): 原始像素坐标 [x, y]。
        img_size (list|None): 图片尺寸 [width, height]。
        gaze_mode (str): 'none' | 'raw' | 'norm'。

    Returns:
        str: 拼接后的 prompt。
    """
    if gaze_mode == 'none' or gaze_point is None:
        return instruction

    gx, gy = gaze_point[0], gaze_point[1]

    if gaze_mode == 'raw':
        prefix = f"I look at [{gx}, {gy}] in this image."
    elif gaze_mode == 'norm':
        if img_size is None:
            
            prefix = f"I look at [{gx}, {gy}] in this image."
        else:
            norm_x = round(gx / img_size[0] * 1000)
            norm_y = round(gy / img_size[1] * 1000)
            prefix = f"I look at [{norm_x}, {norm_y}] in this image."
    else:
        return instruction

    return f"{prefix} {instruction}"


def main(args):
    model = build_model(args)
    print("Load model success")

    
    if args.task_type == 'all':
        filter_task_types = None   # None 表示不过滤
    else:
        filter_task_types = set(t.strip() for t in args.task_type.split(','))
        print(f"task_type 过滤: {filter_task_types}")

    if args.task == "all":
        task_filenames = [
            os.path.splitext(f)[0]
            for f in os.listdir(args.screenspot_test)
            if f.endswith(".json")
        ]
    else:
        task_filenames = args.task.split(",")

    if args.inst_style == "all":
        inst_styles = INSTRUCTION_STYLES
    else:
        inst_styles = args.inst_style.split(",")

    if args.language == "all":
        languages = LANGUAGES
    else:
        languages = args.language.split(",")

    if args.gt_type == "all":
        gt_types = GT_TYPES
    else:
        gt_types = args.gt_type.split(",")

    tasks_to_run = []
    for task_filename in task_filenames:
        dataset = task_filename + ".json"
        with open(os.path.join(args.screenspot_test, dataset), 'r') as f:
            raw_task_data = json.load(f)

        # Normalize new annotation format
        task_data = [_normalize_new_annotation(item) for item in raw_task_data]

        
        task_data = [
            t for t in task_data
            if t.get("instruction") or t.get("instruction_cn")
            if t.get("bbox") is not None
            if t.get("img_size") is not None
            if t.get("img_filename")
        ]

        
        if filter_task_types is not None:
            before = len(task_data)
            task_data = [t for t in task_data if t.get("task_type") in filter_task_types]
            after = len(task_data)
            if before != after:
                print(f"[{task_filename}] task_type 过滤: {before} → {after} 条")

        # Create the list of tasks to run, one item as an instance. Tasks may be reused.
        for inst_style in inst_styles:
            for gt_type in gt_types:
                for lang in languages:
                    for task_instance in task_data:
                        task_instance = copy.deepcopy(task_instance)
                        task_instance["task_filename"] = task_filename
                        task_instance["gt_type"] = gt_type
                        task_instance["instruction_style"] = inst_style
                        task_instance["language"] = lang

                        if lang == "cn":
                            if inst_style != 'instruction' or gt_type != 'positive':
                                raise AttributeError("Only positive samples and 'instruction' style are supported for Chinese instructions.")
                            base_instruction = task_instance.get("instruction_cn")
                        elif lang == "en":
                            base_instruction = task_instance.get("instruction")

                        
                        task_instance["prompt_to_evaluate"] = build_prompt_with_gaze(
                            instruction=base_instruction,
                            gaze_point=task_instance.get("gaze_point"),
                            img_size=task_instance.get("img_size"),
                            gaze_mode=args.gaze_mode,
                        )

                        tasks_to_run.append(task_instance)

    
    if filter_task_types is not None:
        from collections import Counter
        dist = Counter(t.get("task_type") for t in tasks_to_run)
        print(f"过滤后 task_type 分布: {dict(dist)}")

    print(f"Total tasks: {len(tasks_to_run)}")

    results = []
    start_time = time.time()
    correct_count = 0
    processed_count = 0

    pbar = tqdm(tasks_to_run, total=len(tasks_to_run))
    for sample in pbar:
        filename = sample["img_filename"]
        img_path = os.path.join(args.screenspot_imgs, filename)

        if sample["gt_type"] == "positive":
            response = model.ground_only_positive(instruction=sample["prompt_to_evaluate"], image=img_path)
        elif sample["gt_type"] == "negative":
            response = model.ground_allow_negative(instruction=sample["prompt_to_evaluate"], image=img_path)

        
        img_size = sample["img_size"]
        print(f"\n{'='*60}")
        print(f"[DEBUG] id={sample.get('id')}")
        print(f"[DEBUG] instruction: {sample.get('prompt_to_evaluate')}")
        print(f"[DEBUG] img_size: {img_size}")
        print(f"[DEBUG] raw_response: {response.get('raw_response')}")
        print(f"[DEBUG] parsed bbox (normalized 0~1): {response.get('bbox')}")
        print(f"[DEBUG] parsed point (normalized 0~1): {response.get('point')}")

        
        gt_bbox_px = sample.get("bbox")  
        if gt_bbox_px and img_size:
            gt_bbox_norm = [
                gt_bbox_px[0] / img_size[0],
                gt_bbox_px[1] / img_size[1],
                gt_bbox_px[2] / img_size[0],
                gt_bbox_px[3] / img_size[1],
            ]
            print(f"[DEBUG] gt_bbox pixels: {[round(v) for v in gt_bbox_px]}")
            print(f"[DEBUG] gt_bbox normalized: {[round(v, 4) for v in gt_bbox_norm]}")
        else:
            print(f"[DEBUG] gt_bbox pixels: {gt_bbox_px}")

        
        pred_point_norm = response.get("point")
        if pred_point_norm and img_size:
            pred_point_px = [pred_point_norm[0] * img_size[0], pred_point_norm[1] * img_size[1]]
            print(f"[DEBUG] pred_point normalized: {[round(v, 4) for v in pred_point_norm]}")
            print(f"[DEBUG] pred_point pixels: {[round(v) for v in pred_point_px]}")
            
            if gt_bbox_px:
                hit = (gt_bbox_px[0] <= pred_point_px[0] <= gt_bbox_px[2]) and \
                      (gt_bbox_px[1] <= pred_point_px[1] <= gt_bbox_px[3])
                print(f"[DEBUG] hit gt_bbox? {hit}")
        else:
            print(f"[DEBUG] pred_point: None (wrong_format)")
        print(f"{'='*60}\n")
        

        point = response["point"]
        point_in_pixel = [point[0] * img_size[0], point[1] * img_size[1]] if point and img_size else None

        sample_result = {
            "id": sample["id"],
            "img_path": img_path,
            "scenario": sample.get("scenario"),
            "platform": sample.get("platform"),
            "place": sample.get("place"),
            "activity": sample.get("activity"),
            "is_same_window": sample.get("is_same_window"),
            "task_type": sample.get("task_type"),
            "language": sample.get("language"),
            "instruction_style": sample.get("instruction_style"),
            "prompt_to_evaluate": sample.get("prompt_to_evaluate"),
            "gaze_point": sample.get("gaze_point"),   
            "gaze_mode": args.gaze_mode,               
            "gt_type": sample.get("gt_type"),
            "ui_type": sample.get("ui_type"),
            "task_filename": sample.get("task_filename"),
            "pred": point_in_pixel,
            "raw_response": response.get("raw_response"),
        }

        if sample["gt_type"] == "positive":
            correctness = eval_sample_positive_gt(sample, response)
            sample_result.update({"bbox": sample.get("bbox")})
        elif sample["gt_type"] == "negative":
            correctness = eval_sample_negative_gt(sample, response)
        else:
            raise ValueError("Wrong instruction type")

        sample_result.update({"correctness": correctness})
        results.append(sample_result)

        
        processed_count += 1
        if correctness == "correct":
            correct_count += 1
        acc = correct_count / processed_count if processed_count > 0 else 0.0
        elapsed = time.time() - start_time
        avg_per = elapsed / processed_count
        remaining = avg_per * (len(tasks_to_run) - processed_count)

        bbox_px = sample.get("bbox")
        if bbox_px:
            bbox_px_int = [int(round(b)) for b in bbox_px]  # x1,y1,x2,y2
        else:
            bbox_px_int = None

        print(f"[{processed_count}/{len(tasks_to_run)}] id={sample.get('id')} correctness={correctness} acc={acc:.4f} eta={remaining/60:.1f}m bbox_px={bbox_px_int}")
        pbar.set_postfix(acc=f"{acc:.4f}", eta=f"{remaining/60:.1f}m")

    result_report = evaluate(results)
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    with open(args.log_path, 'w') as f:
        json.dump(result_report, f, indent=4)
    logging.info("Evaluation of ScreenSpot finished.")


if __name__ == "__main__":
    main(parse_args())
