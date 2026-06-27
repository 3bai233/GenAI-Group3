import os
import json
import logging
import argparse
import copy
from PIL import Image
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings("ignore")

import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # tensorboard is optional
from transformers import Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
from transformers import Qwen3VLProcessor, Qwen3VLForConditionalGeneration
from transformers import AutoProcessor, AutoModelForVision2Seq

from utils.evaluate import evaluate

def check_point_in_box(point, bbox):
    """check if the point is in the bbox"""
    if not point or not bbox:
        return False
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

def compute_bbox_center(bbox):
    """compute the center point of the bbox"""
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)
from utils.action import Grounding, MaskGrounding, Crop
from utils.pipeline import Pipeline, save_pipeline_json

GT_TYPES = ['positive', 'negative']
INSTRUCTION_STYLES = ['instruction', 'action', 'description']
LANGUAGES = ['en', 'cn']

def perform_gui_grounding_baseline(
        args, screenshot_path, user_query, 
        model,  # Can be Qwen model or UGroundModel instance
        processor,  # None for UGround
        model_type="qwen"  # 🔧 添加 model_type 参数
    ):
    """baseline method: single grounding"""
    # Get prompt_type from args, default to "tianxi"
    prompt_type = getattr(args, 'prompt_type', 'tianxi')
    
    pipeline = Pipeline(
        action_dict={
            1: {"input": 0, "action": Grounding(prompt_type=prompt_type, model_type=model_type)}
        }
    )
    pipeline.initiate(
        image=Image.open(screenshot_path), user_query=user_query,
        model=model, processor=processor
    )
    action_list = pipeline.get_actions()

    for index, action in action_list:
        input_dict = pipeline.get_input(action["input"])
        output_dict = action["action"](
            input_dict=input_dict, 
            model_dict=pipeline.model
        )
        pipeline.update(index, output_dict)
        logging.info(f"Pipeline: {index} / {len(action_list)}")

    # add baseline information
    final_result = pipeline.get_final_result()
    final_result["baseline_bbox"] = output_dict.get("bbox_abs")
    final_result["baseline_point"] = output_dict.get("point_abs")
    
    return final_result, pipeline


def perform_gui_grounding_with_reground_judge_two_images(
        args, screenshot_path, user_query, 
        model,  # Can be Qwen model or UGroundModel instance
        processor,  # None for UGround
        judge_action=None,  # 🔧 接收预初始化的 judge action
        prompt_type="tianxi",  # 🔧 添加 prompt_type 参数
        model_type="qwen"  # 🔧 添加 model_type 参数 (qwen or uground)
    ):
    """use ReGrounding+DrawDualBoxesSeparate+JudgeTwoImages to design the pipeline"""
    from utils.action import DrawDualBoxesSeparate, GPTJudgeTwoImages
    
    # 🔧 Use pre-initialized judge action (avoid reloading model each time)
    if judge_action is None:
        # Fallback: create new instance (for backward compatibility)
        if args.use_local_judge:
            from utils.action_local_judge import LocalJudgeTwoImages
            logging.info(f"Creating LOCAL judge model: {args.local_model_path}")
            judge_action = LocalJudgeTwoImages(
                model_path=args.local_model_path
            )
        else:
            logging.info(f"Creating GPT judge: {args.gpt_model}")
            judge_action = GPTJudgeTwoImages(
                api_key=args.gpt_api_key,
                base_url=args.gpt_base_url,
                model=args.gpt_model
            )
    
    pipeline = Pipeline(
        action_dict={
            1: {"input": 0, "action": Grounding(prompt_type=prompt_type, model_type=model_type)},
            2: {"input": [0,1], "action": Crop(0.2)},
            3: {"input": 2, "action": MaskGrounding(prompt_type=prompt_type, model_type=model_type)},    
            4: {"input": 3, "action": Grounding(prompt_type=prompt_type, model_type=model_type)},
            5: {"input": [3, 4], "action": DrawDualBoxesSeparate()},
            6: {"input": 5, "action": judge_action}  # 🔧 Use selected judge method
        }
    )
    pipeline.initiate(
        image=Image.open(screenshot_path), user_query=user_query,
        model=model, processor=processor
    )
    action_list = pipeline.get_actions()

    for index, action in action_list:
        input_dict = pipeline.get_input(action["input"])
        output_dict = action["action"](
            input_dict=input_dict, 
            model_dict=pipeline.model
        )
        pipeline.update(index, output_dict)
        logging.info(f"Pipeline: {index} / {len(action_list)}")
    
    final_result = pipeline.get_final_result()
    
    return final_result, pipeline

def eval_sample_positive_gt(sample, point_pred):
    bbox = sample["bbox"]
    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]  # x1, y1, x2, y2
    
    if point_pred is None or len(point_pred) != 2:
        correctness = "wrong_format"
    elif (bbox[0] <= point_pred[0] <= bbox[2]) and (bbox[1] <= point_pred[1] <= bbox[3]):
        correctness = "correct"
    else:
        correctness = "wrong"
    return correctness

def build_uid(sample: dict) -> str:
    return f"{sample.get('task_filename')}-{sample.get('img_filename')}-{sample.get('language')}|{sample.get('instruction_style')}|{sample.get('gt_type')}"

def load_partial_state(state_path: str):
    if state_path and os.path.exists(state_path):
        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            results = state.get("results", [])
            return results
        except Exception as e:
            logging.warning("Failed to load resume state: %s", e)
    return []

def save_partial_state(state_path: str, results: list):
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    tmp_path = state_path + ".tmp"
    payload = {"results": results}
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, state_path)


def main(args, writer):

    # load model
    logging.info("using torch inference...")
    
    if args.model_type == "uground":
        # UGround 使用 LLaVA 架构，需要特殊加载方式
        logging.info(f"Loading UGround model from: {args.model_path}")
        from utils.models.uground import UGroundModel
        uground_model = UGroundModel()
        uground_model.load_model(args.model_path)
        model = uground_model  # 传递 UGroundModel 实例
        processor = None  # UGround 不使用 processor
        logging.info(f"✓ Loaded UGround model: {args.model_path}")
    else:
        # Qwen 系列模型 (包括 Holo2)
        # 检测是否为 30B 模型，30B 需要多卡并行
        model_path_lower = str(args.model_path).lower()
        is_30b = "30b" in model_path_lower
        grounding_attn_implementation = os.environ.get("GROUNDING_ATTN_IMPLEMENTATION", "eager")
        grounding_min_pixels = int(os.environ.get("GROUNDING_MIN_PIXELS", str(256 * 28 * 28)))
        grounding_max_pixels = int(os.environ.get("GROUNDING_MAX_PIXELS", str(1024 * 28 * 28)))
        logging.info(f"Grounding attention implementation: {grounding_attn_implementation}")
        logging.info(f"Grounding processor pixels: min={grounding_min_pixels}, max={grounding_max_pixels}")
        
        if is_30b:
            # 30B 模型：使用 device_map="auto" 自动分配到多卡
            logging.info(f"Detected 30B model, using multi-GPU with device_map='auto'")
            try:
                from transformers import AutoModelForImageTextToText
                model = AutoModelForImageTextToText.from_pretrained(
                    args.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",  # 自动分配到多卡
                    trust_remote_code=True
                )
                processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, min_pixels=grounding_min_pixels, max_pixels=grounding_max_pixels)
                logging.info(f"✓ Loaded 30B model with multi-GPU: {model.__class__.__name__}")
                logging.info(f"✓ Loaded processor: {processor.__class__.__name__}")
            except Exception as e:
                logging.error(f"Failed to load 30B model: {e}")
                raise
        else:
            # 小模型（8B等）：使用单卡
            try:
                model = AutoModelForVision2Seq.from_pretrained(
                    args.model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation=grounding_attn_implementation,
                    device_map="cuda:0",  # 单卡
                    trust_remote_code=True
                )
                processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, min_pixels=grounding_min_pixels, max_pixels=grounding_max_pixels)
                logging.info(f"✓ Loaded model using AutoModel: {model.__class__.__name__}")
                logging.info(f"✓ Loaded processor: {processor.__class__.__name__}")
            except Exception as e:
                logging.error(f"Failed to load model with AutoModel, trying Qwen2.5: {e}")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    args.model_path,
                    torch_dtype=torch.bfloat16,
                    attn_implementation=grounding_attn_implementation,
                    device_map="cuda:0"
                )        
                processor = Qwen2_5_VLProcessor.from_pretrained(args.model_path, min_pixels=grounding_min_pixels, max_pixels=grounding_max_pixels)

    # load ss-pro tasks
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
        with open(os.path.join(args.screenspot_test, dataset), 'r', encoding='utf-8') as f:
            task_data = json.load(f)

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
                            task_instance["prompt_to_evaluate"] = task_instance["instruction_cn"]
                        else:
                            task_instance["prompt_to_evaluate"] = task_instance["instruction"]

                        tasks_to_run.append(task_instance)
        logging.info(f"Num of sample in {task_filename}: {len(task_data)} * {len(inst_styles)} * {len(gt_types)} * {len(languages)} = {len(task_data) * len(inst_styles) * len(gt_types) * len(languages)}")
    logging.info(f"Total tasks: {len(tasks_to_run)}")

    state_path = args.resume_state if args.resume_state else os.path.join(args.root_path, f"{args.log_name}.state.json")
    results = []
    processed_uids = set()
    if args.resume:
        results = load_partial_state(state_path)
        processed_uids = {r.get("uid") for r in results if r.get("uid")}
        logging.info(f"Resume enabled. Loaded %d finished samples. {len(processed_uids)}")

    corr_action = sum(1 for r in results if r.get("correctness") == "correct")
    num_action = len(results)
    indices_to_run = [i for i in range(len(tasks_to_run)) if build_uid(tasks_to_run[i]) not in processed_uids]
    pbar = tqdm(indices_to_run, total=len(tasks_to_run), initial=num_action, desc="Evaluating")

    # 🔧 Initialize judge action once (avoid reloading model for each sample)
    judge_action_instance = None
    if args.use_reground_judge_two_images:
        if args.use_local_judge:
            from utils.action_local_judge import LocalJudgeTwoImages
            logging.info(f"Initializing LOCAL judge model (once): {args.local_model_path}")
            judge_action_instance = LocalJudgeTwoImages(
                model_path=args.local_model_path
            )
            logging.info(f"✓ Local judge model ready")
        elif args.use_api_judge:
            # Use API-based judge (OpenAI, Gemini, or Gemini Thinking)
            from utils.action_api_judge import APIJudgeTwoImages
            logging.info(f"Initializing API judge: type={args.judge_api_type}, model={args.gpt_model}")
            
            # Determine API key based on type
            if args.judge_api_type.startswith("gemini"):
                api_key = args.gemini_api_key
            else:
                api_key = args.gpt_api_key
            
            judge_action_instance = APIJudgeTwoImages(
                api_type=args.judge_api_type,
                api_key=api_key,
                base_url=args.gpt_base_url if args.gpt_base_url != "None" else None,
                model=args.gpt_model if args.gpt_model != "None" else None,
                thinking_budget=args.thinking_budget
            )
            logging.info(f"✓ API judge ready: {args.judge_api_type}")

    # start inference
    for index in pbar:
        sample = tasks_to_run[index]
        uid = build_uid(sample)
        filename = sample["img_filename"]
        img_path = os.path.join(args.screenspot_imgs, filename)
        user_query = sample["prompt_to_evaluate"]

        logging.info("--------------------------------------------------------")
        logging.info(f"img_path: {img_path}")
        logging.info(f"user_query: {user_query}")

        try:
            # according to the parameters, choose the pipeline to use
            if args.use_reground_judge_two_images:
                output_dict, pipeline = perform_gui_grounding_with_reground_judge_two_images(
                    args, img_path, user_query, model, processor,
                    judge_action=judge_action_instance,  # 🔧 传递预初始化的 judge
                    prompt_type=args.prompt_type,  # 🔧 传递 prompt_type
                    model_type=args.model_type  # 🔧 传递 model_type
                )
            else:
                output_dict, pipeline = perform_gui_grounding_baseline(args, img_path, user_query, model, processor, 
                                                                        model_type=args.model_type)
            save_pipeline_json(pipeline, args.pipeline_dir, f"{index}")
            
            output_text, point_pred = output_dict["output_text"], output_dict["point_abs"]
            
            # If using GPT judge, log detailed results
            if args.use_reground_judge_two_images:
                logging.info("=== GPT Judge Result Summary ===")
                logging.info(f"User query: {user_query}")
                logging.info(f"Final selection: Box {output_dict.get('selected_box', output_dict.get('selected_image', 'N/A'))}")
                logging.info(f"Final coordinates: {point_pred}")
                if 'judge_reason' in output_dict:
                    # Show only first 500 characters to avoid overly long logs
                    reason_preview = output_dict['judge_reason'][:500]
                    if len(output_dict['judge_reason']) > 500:
                        reason_preview += "..."
                    logging.info(f"Judge reason:\n{reason_preview}")
                logging.info("====================")

            sample_result = {
                "uid": uid,
                "img_path": img_path,
                "group": sample["group"] if "group" in sample else None,
                "platform": sample["platform"],
                "application": sample["application"],
                "language": sample["language"],
                "instruction_style": sample["instruction_style"],
                "prompt_to_evaluate": sample["prompt_to_evaluate"],
                "gt_type": sample["gt_type"],
                "ui_type": sample["ui_type"],
                "task_filename": sample["task_filename"],
                "pred": point_pred,
                "raw_response": output_text,
                "gt": sample["bbox"]
            }
            
            # If using reground_judge_two_images, add relevant fields
            if args.use_reground_judge_two_images:
                # 支持两种字段名: selected_box 或 selected_image
                selected = output_dict.get("selected_box") or output_dict.get("selected_image")
                sample_result.update({
                    "selected_box": selected,
                    "judge_reason": output_dict.get("judge_reason"),
                    "judge_response": output_dict.get("judge_response"),
                    "bbox1": output_dict.get("bbox1"),
                    "bbox2": output_dict.get("bbox2")
                })
                
            # Add baseline information (if exists)
            if "baseline_bbox" in output_dict:
                sample_result["baseline_bbox"] = output_dict["baseline_bbox"]
                sample_result["baseline_point"] = output_dict["baseline_point"]

            correctness = eval_sample_positive_gt(sample, point_pred)

            num_action += 1
            if correctness == "correct":
                corr_action += 1
            
            # Basic result log
            result_log = f"Result: {correctness} | Acc: {corr_action / num_action:.2f} | Pred: {point_pred} | GT: {sample['bbox']}"
            
            # Add judgment information if using judge
            if args.use_reground_judge_two_images and selected:
                judge_reason = output_dict.get("judge_reason", "")[:50] + "..." if len(output_dict.get("judge_reason", "")) > 50 else output_dict.get("judge_reason", "")
                result_log += f" | Judge Choice: Box{selected} | Reason: {judge_reason}"
            
            logging.info(result_log)
            writer.add_scalar("Acc", corr_action / num_action, num_action)
            sample_result.update({"correctness": correctness})
            results.append(sample_result)

            save_partial_state(state_path, results)
            pbar.set_postfix(acc=f"{corr_action / max(1,num_action):.3f}")

        except Exception as e:
            save_partial_state(state_path, results)
            logging.exception("Error occurred at index %d (%s). Partial state saved to %s", index, uid, state_path)
            raise e

    result_report = evaluate(results)


    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    with open(args.log_path, 'w', encoding='utf-8') as f:
        json.dump(result_report, f, ensure_ascii=False, indent=4)
    logging.info(f"Evaluation of ScreenSpot finished. saved to {args.log_path}")

if __name__ == "__main__":
    start_time = time.time()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type=str, default="/path/to/your/model/")
    parser.add_argument('--screenspot_imgs', type=str, default="/path/to/ScreenSpot-Pro/images")
    parser.add_argument('--screenspot_test', type=str, default="/path/to/ScreenSpot-Pro/annotations")

    parser.add_argument('--task', type=str, default="all")
    parser.add_argument('--inst_style', type=str, default="instruction", choices=INSTRUCTION_STYLES + ['all'], help="Instruction style to use.")
    parser.add_argument('--language', type=str, default="en" , choices=LANGUAGES + ['all'], help="Language to use.")
    parser.add_argument('--gt_type', type=str, default="positive", choices=GT_TYPES + ['all'], help="Ground truth type: 'positive' or 'negative'.")

    parser.add_argument('--root_path', type=str, default="./eval_results")
    parser.add_argument("--log_name", type=str, default="test")

    parser.add_argument('--resume', action='store_true', help="Enable resume from partial state file.")
    parser.add_argument('--resume_state', type=str, default=None, help="Path to partial state file. Default: <log_path>.state.json")
    parser.add_argument('--pipeline_dir', type=str, default=None,
                    help="Directory to save per-sample pipeline json files. Default: <root_path>/pipelines")
    
    # ReGrounding judgment related parameters
    parser.add_argument('--use_reground_judge_two_images', action='store_true',
                        help='Use ReGrounding+DrawDualBoxesSeparate+GPTJudgeTwoImages combined pipeline')
    parser.add_argument('--gpt_api_key', type=str, default=None,
                        help='OpenAI API key')
    parser.add_argument('--gpt_base_url', type=str, default="None",
                        help='OpenAI API base URL')
    parser.add_argument('--gpt_model', type=str, default="None",
                        help='model name to use')
    
    # Local judge model (finetuned Qwen3-VL) parameters
    parser.add_argument('--use_local_judge', action='store_true',
                        help='Use finetuned local Qwen3-VL model instead of GPT')
    parser.add_argument('--local_model_path', type=str,
                        default='/data1/model_checkpoint_GUI/two_box_judge_qwen3_8b/checkpoints/checkpoint-2712',
                        help='Path to finetuned local model checkpoint (use latest available)')
    
    # API Judge parameters (supports OpenAI-compatible and Gemini APIs)
    parser.add_argument('--use_api_judge', action='store_true',
                        help='Use API-based judge (OpenAI, Gemini, or Gemini Thinking)')
    parser.add_argument('--judge_api_type', type=str, default='openai',
                        choices=['openai', 'gemini', 'gemini_thinking'],
                        help='API type: openai (OpenAI-compatible), gemini (standard), gemini_thinking (with thinking)')
    parser.add_argument('--gemini_api_key', type=str, default=None,
                        help='Gemini API key (can also use GEMINI_API_KEY env var)')
    parser.add_argument('--thinking_budget', type=int, default=8192,
                        help='Thinking token budget for Gemini thinking models')
    
    # Prompt type selection
    parser.add_argument('--prompt_type', type=str, default='tianxi', 
                        choices=['tianxi', 'qwen3', 'vlmevalkit', 'holo2_json', 'qwen25vl_toolcall', 'uground'],
                        help='Prompt format: tianxi (original), qwen3 (function calling), vlmevalkit (pyautogui), holo2_json (official JSON format), qwen25vl_toolcall (Qwen2.5-VL tool_call format), uground (UGround LLaVA format)')
    
    # Model type selection (for different model architectures)
    parser.add_argument('--model_type', type=str, default='qwen', 
                        choices=['qwen', 'uground'],
                        help='Model type: qwen (Qwen/Holo2 models), uground (LLaVA-based UGround)')

    args = parser.parse_args()
    os.makedirs(args.root_path, exist_ok=True)
    args.log_path = os.path.join(args.root_path, f"{args.log_name}.json")
    if args.pipeline_dir is None:
        args.pipeline_dir = os.path.join(args.root_path, args.log_name, "pipelines")

    writer = SummaryWriter(log_dir=os.path.join(args.root_path, args.log_name)) if SummaryWriter is not None else None

    logging.info("args: %s", args)
    main(args, writer)

    end_time = time.time()
    duration_min = (end_time - start_time) / 60
    logging.info(f"inference completed, taking {duration_min} mins")
