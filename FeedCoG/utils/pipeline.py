import os
import json

def _is_primitive(v):
    return isinstance(v, (str, int, float, bool)) or v is None

def _filter_for_json(obj):
    if _is_primitive(obj):
        return obj
    if isinstance(obj, (list, tuple)):
        kept = []
        for x in obj:
            fx = _filter_for_json(x)
            if _is_primitive(fx):
                kept.append(fx)
        return kept
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            fv = _filter_for_json(v)
            if _is_primitive(fv):
                out[k] = fv
            elif isinstance(fv, list) and all(_is_primitive(i) for i in fv):
                out[k] = fv
        return out
    return None

def save_pipeline_json(pipeline, save_dir, uid):

    os.makedirs(save_dir, exist_ok=True)
    filtered_results = {}
    for step_id, content in getattr(pipeline, "results", {}).items():
        filtered = _filter_for_json(content)
        if isinstance(filtered, dict) and filtered:
            filtered_results[str(step_id)] = filtered

    payload = {
        "uid": uid,
        "results": filtered_results
    }
    out_path = os.path.join(save_dir, f"{uid}.pipeline.json")
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, out_path)

class Pipeline:
    def __init__(self, action_dict, output_id=None):
        self.action = action_dict
        if output_id is None:
            self.output_id = max(action_dict.keys())
        else:
            self.output_id = output_id
    
    def get_final_result(self):
        return self.results[self.output_id]
    
    def get_actions(self):
        action_list = sorted([(k, v) for (k, v) in self.action.items()], key=lambda x: x[0])
        return action_list
    
    def initiate(self, image, user_query, model, processor):
        self.results = {
            0: {"image": image, "coord_abs": None, 
                "user_query": user_query}
        }
        self.model = {
            "model": model,
            "processor": processor,
            "original_image": image  # 保存原始图像供后续使用
        }
    
    def update(self, idx, content):
        assert "image" in content, f"image not found in {content}"
        content["pipeline_id"] = idx
        self.results[idx] = content
    
    def get_input(self, input_id):
        if isinstance(input_id, list):
            return [self.results[i] for i in input_id]
        else:
            return [self.results[input_id]]