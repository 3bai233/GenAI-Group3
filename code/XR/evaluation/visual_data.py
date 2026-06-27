import os
import json
import math
import argparse
from collections import defaultdict, Counter
from PIL import Image, ImageDraw


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_text(x):
    if x is None:
        return "None"
    s = str(x)
    return s.replace("\n", " ").replace("|", "\\|")


def load_result(result_json):
    with open(result_json, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_image_path(sample, images_dir):
    img_path = sample.get("img_path")
    if img_path and os.path.exists(img_path):
        return img_path
    if img_path:
        basename = os.path.basename(img_path)
        p = os.path.join(images_dir, basename)
        if os.path.exists(p):
            return p
    return None


def draw_sample_vis(sample, images_dir, out_path):
    img_path = resolve_image_path(sample, images_dir)
    if not img_path or not os.path.exists(img_path):
        return False

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    bbox = sample.get("bbox")
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=8)

    pred = sample.get("pred")
    if pred and len(pred) == 2:
        px, py = [int(round(v)) for v in pred]
        r = 30
        draw.ellipse([px - r, py - r, px + r, py + r], outline=(255, 0, 0), width=8)

    err = sample.get("_error_type")
    txt = f"id={sample.get('id')} correctness={sample.get('correctness')}"
    if err:
        txt += f" error={err}"
    draw.rectangle([10, 10, 10 + 12 * len(txt), 38], fill=(0, 0, 0))
    draw.text((14, 14), txt, fill=(255, 255, 255))

    img.save(out_path)
    return True


def metric_table_md(metrics_dict):
    lines = []
    lines.append("| key | num_correct | num_total | acc | text_acc | icon_acc |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for k, v in metrics_dict.items():
        lines.append(
            f"| {safe_text(k)} | {v.get('num_correct_action', 0)} | {v.get('num_total', 0)} | "
            f"{v.get('action_acc', 0):.4f} | {v.get('text_acc', 0):.4f} | {v.get('icon_acc', 0):.4f} |"
        )
    return "\n".join(lines)


def get_field(d, key):
    if key == "task_type":
        return d.get("task_type", d.get("task type"))
    return d.get(key)


def group_acc(details, key):
    stat = defaultdict(lambda: [0, 0])  # correct, total
    for d in details:
        k = get_field(d, key)
        stat[k][1] += 1
        if d.get("correctness") == "correct":
            stat[k][0] += 1

    rows = sorted(stat.items(), key=lambda x: x[1][1], reverse=True)
    lines = []
    lines.append(f"| {key} | correct | total | acc |")
    lines.append("|---|---:|---:|---:|")
    for k, (c, t) in rows:
        acc = c / t if t else 0
        lines.append(f"| {safe_text(k)} | {c} | {t} | {acc:.4f} |")
    return "\n".join(lines)


def classify_error_type(sample, images_dir, near_miss_thresh=0.1, wrong_element_offset_thresh=0.3):
    bbox = sample.get("bbox")
    pred = sample.get("pred")
    if not bbox or len(bbox) != 4 or not pred or len(pred) != 2:
        return {
            "error_type": "invalid_pred_or_bbox",
            "norm_offset_x": None,
            "norm_offset_y": None,
            "norm_distance": None,
        }

    img_path = resolve_image_path(sample, images_dir)
    if not img_path or not os.path.exists(img_path):
        return {
            "error_type": "invalid_image_path",
            "norm_offset_x": None,
            "norm_offset_y": None,
            "norm_distance": None,
        }

    try:
        with Image.open(img_path) as im:
            img_w, img_h = im.size
    except Exception:
        return {
            "error_type": "invalid_image_size",
            "norm_offset_x": None,
            "norm_offset_y": None,
            "norm_distance": None,
        }

    x1, y1, x2, y2 = [float(v) for v in bbox]
    px, py = [float(v) for v in pred]

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    offset_x = px - center_x
    offset_y = py - center_y

    nw = max(float(img_w), 1e-6)
    nh = max(float(img_h), 1e-6)

    norm_offset_x = offset_x / nw
    norm_offset_y = offset_y / nh
    norm_distance = math.sqrt(norm_offset_x ** 2 + norm_offset_y ** 2)

    if norm_distance < near_miss_thresh:
        error_type = "near_miss"
    elif abs(norm_offset_x) > wrong_element_offset_thresh or abs(norm_offset_y) > wrong_element_offset_thresh:
        error_type = "wrong_element"
    else:
        error_type = "moderate_offset"

    return {
        "error_type": error_type,
        "norm_offset_x": norm_offset_x,
        "norm_offset_y": norm_offset_y,
        "norm_distance": norm_distance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_json", 
                        default="path/to/your/results",
                        type=str, required=True, help="Path to evaluation results JSON")
    parser.add_argument(
        "--images_dir",
        type=str,
        default="path/to/your/data",
        help="Path to dataset images directory",
    )
    parser.add_argument("--output_md", type=str, required=True, help="Path to output markdown file")
    parser.add_argument("--vis_dir", type=str, default=None, help="Path to visualization images directory")
    parser.add_argument("--max_wrong_cases", type=int, default=1000, help="Maximum number of wrong cases to visualize")

    parser.add_argument("--near_miss_thresh", type=float, default=0.1, help="Near miss threshold (normalized distance)")
    parser.add_argument(
        "--wrong_element_offset_thresh",
        type=float,
        default=0.3,
        help="Wrong element threshold (normalized offset_x/offset_y absolute value)",
    )
    parser.add_argument("--target_ids", nargs="+", type=str, default=None, help="Visualize only specified sample IDs and exit, e.g: --target_ids 1 2 3")
    args = parser.parse_args()

    data = load_result(args.result_json)
    details = data.get("details", [])
    metrics = data.get("metrics", {})

    output_md = os.path.abspath(args.output_md)
    out_root = os.path.dirname(output_md)
    ensure_dir(out_root)

    if args.vis_dir is None:
        vis_dir = os.path.join(out_root, "vis_cases")
    else:
        vis_dir = os.path.abspath(args.vis_dir)
    ensure_dir(vis_dir)

    if args.target_ids is not None:
        target_samples = [d for d in details if str(d.get("id")) in args.target_ids]
        if not target_samples:
            return
            
        for target_sample in target_samples:
            err_info = classify_error_type(
                target_sample,
                args.images_dir,
                near_miss_thresh=args.near_miss_thresh,
                wrong_element_offset_thresh=args.wrong_element_offset_thresh,
            )
            target_sample["_error_type"] = err_info["error_type"]
            
            base = os.path.basename(target_sample.get("img_path", "unknown.png"))
            vis_name = f"id_{target_sample.get('id')}_{base}"
            vis_path = os.path.join(vis_dir, vis_name)
            
            ok = draw_sample_vis(target_sample, args.images_dir, vis_path)
            if ok:
                print(f"\n[OK] Successfully visualized sample {target_sample.get('id')}, saved to: {vis_path}")
                print(f" -> prompt: {target_sample.get('prompt_to_evaluate')}")
                print(f" -> bbox (GT): {target_sample.get('bbox')}")
                print(f" -> pred (Model): {target_sample.get('pred')}")
                print(f" -> correctness: {target_sample.get('correctness')} ({err_info['error_type']})")
                print(f" -> raw_response: {target_sample.get('raw_response')}")
            else:
                print(f"\n[ERROR] Failed to generate image for sample {target_sample.get('id')}, please check if image exists: {resolve_image_path(target_sample, args.images_dir)}")
        return

    wrong_cases = [d for d in details if d.get("correctness") != "correct"][: args.max_wrong_cases]
    vis_rel_paths = []
    for d in wrong_cases:
        err_info = classify_error_type(
            d,
            args.images_dir,
            near_miss_thresh=args.near_miss_thresh,
            wrong_element_offset_thresh=args.wrong_element_offset_thresh,
        )
        d["_error_type"] = err_info["error_type"]
        d["_norm_offset_x"] = err_info["norm_offset_x"]
        d["_norm_offset_y"] = err_info["norm_offset_y"]
        d["_norm_distance"] = err_info["norm_distance"]

        base = os.path.basename(d.get("img_path", "unknown.png"))
        vis_name = f"id_{d.get('id')}_{base}"
        vis_path = os.path.join(vis_dir, vis_name)
        ok = draw_sample_vis(d, args.images_dir, vis_path)
        if ok:
            rel = os.path.relpath(vis_path, out_root)
            vis_rel_paths.append((d, rel))

    ERROR_TYPE_ORDER = {"near_miss": 0, "moderate_offset": 1, "wrong_element": 2}

    def sort_key(item):
        d, _ = item
        scenario = safe_text(get_field(d, "scenario"))
        activity = safe_text(get_field(d, "activity"))
        error_type = d.get("_error_type", "unknown")
        error_order = ERROR_TYPE_ORDER.get(error_type, 99)
        return (scenario, activity, error_order)

    vis_rel_paths.sort(key=sort_key)

    lines = []
    lines.append("# GUI Grounding Evaluation Report")
    lines.append("")
    lines.append(f"- result_json: `{os.path.abspath(args.result_json)}`")
    lines.append(f"- images_dir: `{os.path.abspath(args.images_dir)}`")
    lines.append(f"- total_samples: **{len(details)}**")
    lines.append(f"- near_miss_thresh: **{args.near_miss_thresh}**")
    lines.append(f"- wrong_element_offset_thresh: **{args.wrong_element_offset_thresh}**")
    lines.append("")

    overall = metrics.get("overall", {})
    lines.append("## Overall")
    lines.append("")
    lines.append(
        f"- num_correct: **{overall.get('num_correct_action', 0)}** / **{overall.get('num_total', 0)}**"
    )
    lines.append(f"- action_acc: **{overall.get('action_acc', 0):.4f}**")
    lines.append(f"- text_acc: **{overall.get('text_acc', 0):.4f}**")
    lines.append(f"- icon_acc: **{overall.get('icon_acc', 0):.4f}**")
    lines.append("")

    lines.append("## Error Type Statistics (Wrong Cases)")
    lines.append("")
    err_counter = Counter(d.get("_error_type", "unknown") for d in wrong_cases)
    lines.append("| error_type | count | ratio |")
    lines.append("|---|---:|---:|")
    total_wrong = len(wrong_cases)
    for et, c in sorted(err_counter.items(), key=lambda x: x[1], reverse=True):
        ratio = c / total_wrong if total_wrong else 0
        lines.append(f"| {safe_text(et)} | {c} | {ratio:.4f} |")
    lines.append("")

    lines.append("## Task Type Accuracy")
    lines.append("")
    lines.append(group_acc(details, "task_type"))
    lines.append("")

    lines.append("## Group Statistics (from details)")
    lines.append("")
    for k in ["scenario", "place", "activity", "platform", "task_type", "ui_type", "is_same_window"]:
        lines.append(f"### by {k}")
        lines.append("")
        lines.append(group_acc(details, k))
        lines.append("")

    lines.append("## Metrics Sections (from json.metrics)")
    lines.append("")
    for sec in ["fine_grained", "seeclick_style", "leaderboard_simple_style", "leaderboard_detailed_style"]:
        sec_data = metrics.get(sec, {})
        if not sec_data:
            continue
        lines.append(f"### {sec}")
        lines.append("")
        lines.append(metric_table_md(sec_data))
        lines.append("")

    lines.append("## Wrong Cases Visualization")
    lines.append("")
    lines.append("> Green box: GT bbox; Red dot: Predicted point")
    lines.append("")

    cur_scenario = None
    cur_activity = None
    for d, rel in vis_rel_paths:
        scenario = safe_text(get_field(d, "scenario"))
        activity = safe_text(get_field(d, "activity"))

        if scenario != cur_scenario:
            cur_scenario = scenario
            cur_activity = None
            lines.append(f"## Scenario: {scenario}")
            lines.append("")

        if activity != cur_activity:
            cur_activity = activity
            lines.append(f"### Activity: {activity}")
            lines.append("")

        lines.append(f"#### id={d.get('id')} | {safe_text(d.get('prompt_to_evaluate'))}")
        lines.append("")

        lines.append(f"- scenario: `{safe_text(get_field(d, 'scenario'))}`")
        lines.append(f"- platform: `{safe_text(get_field(d, 'platform'))}`")
        lines.append(f"- place: `{safe_text(get_field(d, 'place'))}`")
        lines.append(f"- activity: `{safe_text(get_field(d, 'activity'))}`")
        lines.append(f"- is_same_window: `{safe_text(get_field(d, 'is_same_window'))}`")
        lines.append(f"- task_type: `{safe_text(get_field(d, 'task_type'))}`")
        lines.append(f"- ui_type: `{safe_text(get_field(d, 'ui_type'))}`")

        lines.append(f"- error_type: `{safe_text(d.get('_error_type'))}`")
        lines.append(f"- norm_offset_x: `{safe_text(d.get('_norm_offset_x'))}`")
        lines.append(f"- norm_offset_y: `{safe_text(d.get('_norm_offset_y'))}`")
        lines.append(f"- norm_distance: `{safe_text(d.get('_norm_distance'))}`")

        lines.append(f"- pred: `{safe_text(d.get('pred'))}`")
        lines.append(f"- bbox: `{safe_text(d.get('bbox'))}`")
        lines.append(f"- raw_response: `{safe_text(d.get('raw_response'))}`")
        lines.append("")
        lines.append(f"![id_{d.get('id')}]({rel})")
        lines.append("")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] markdown saved to: {output_md}")
    print(f"[OK] vis images dir: {vis_dir}")


if __name__ == "__main__":
    main()