# `UITarsBamiAgent` — UI-TARS 1.5 + BAMI 动作精炼 Agent

**文件：** `/share/home/group3/agent/OSWorld/uitars15_v2_bami.py`

## 概览

`UITarsBamiAgent` 继承自 `UITarsAgent`（来自 `mm_agents.uitars15_v2`），在 UI-TARS 1.5 视觉语言模型的基础上，引入了 **BAMI（Bounding-box Augmented Model Improvement）** 动作精炼机制。Agent 通过一个本地视觉语言模型（Judge Model）对 UI-TARS 的原始预测坐标进行二次校验和修正，提升 GUI 操作的点击精度。

## 核心流程

```
UI-TARS 1.5 VLM（Doubao / vLLM）
    │
    ▼
原始预测（action_type + bbox 坐标）
    │
    ▼
┌─── BAMI 精炼 ──────────────────────────┐
│  1. Judge 模型分析截图 + 原始预测        │
│  2. 判断是否需要重定位                   │
│  3. 如需要：裁剪目标区域 → 重新预测       │
│  4. 选择最优结果（原始 vs 重定位）       │
└─────────────────────────────────────────┘
    │
    ▼
最终 pyautogui 代码 → env.step()
```

## 类结构

### `__init__` 初始化

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | str | — | UI-TARS 模型 ID（Doubao API / vLLM） |
| `model_type` | str | — | `doubao` / `qwen25` / `qwen25vl` |
| `enable_bami` | bool | `False` | 是否启用 BAMI 精炼 |
| `bami_local_judge_model_path` | str | `None` | 本地 Judge 模型路径 |
| `bami_local_judge_base_model_path` | str | `None` | 本地 Judge 基座模型路径 |
| `bami_local_judge_gpu` | str | `None` | Judge 模型 GPU 设备号 |
| `bami_mask_ratio` | float | `0.12` | 目标区域裁剪的 mask 比例 |
| `bami_crop_expand_ratio` | float | `0.2` | 目标区域裁剪的扩展比例 |
| `max_steps` | int | `100` | 最大步数 |
| `use_thinking` | bool | `True` | 是否启用 Thought 推理字段 |
| `language` | str | `"Chinese"` | 输出语言 |

> 其他参数（`max_tokens`, `top_p`, `temperature`, `max_trajectory_length`, `max_image_history_length` 等）继承自 `UITarsAgent`。

### 主要方法

#### `_feedcog_root() -> str`
定位 `FeedCoG` 项目根目录。搜索顺序：
1. 环境变量 `FEEDCOG_ROOT`
2. 当前文件目录向上遍历
3. `os.getcwd()` 向上遍历
4. 回退到 `../../FeedCoG`

#### `_try_create_local_judge() -> Optional[object]`
尝试加载本地 Judge 模型（vLLM）。Judge 模型用于对 UI-TARS 的预测进行二次校验。如果指定了 `bami_local_judge_model_path`，则通过 vLLM 加载该模型；否则加载 `bami_local_judge_base_model_path` 并应用 LoRA 适配器。

#### `_judge(raw_prediction, reground_prediction, instruction, source_image, image_width, image_height) -> dict`
**Judge 核心方法**。比较 UI-TARS 的原始预测和重定位预测，判断哪个更优：
- 构造 Judge prompt，包含任务指令、原始动作、裁剪后截图
- Judge 模型输出 JSON 格式的判断结果：`{ "correct": bool, "reason": str, "selected": "baseline" | "reground" }`
- 返回详细的判决信息

#### `_predict_with_reground(task_instruction, source_image, candidate, image_width, image_height) -> dict`
对指定的候选动作进行**重定位预测**：
1. 根据候选动作的 bbox 坐标裁剪截图
2. 将裁剪区域作为输入发给 UI-TARS
3. UI-TARS 重新预测目标区域的精确坐标
4. 将裁剪坐标映射回原图坐标

#### `_bbox_to_action_box(box_key, bbox, image_width, image_height) -> dict`
将 bbox 坐标转换为动作输入格式。

#### `_maybe_refine_with_bami(task_instruction, source_image, prediction, parsed_dict, image_width, image_height) -> tuple`
**BAMI 精炼入口**。对 `predict()` 方法输出的候选动作列表逐一检查：
- 如果 BAMI 未启用 → 直接返回原始结果
- 如果 BAMI 启用 → 对每个候选动作：
  1. 调用 `_predict_with_reground()` 进行重定位预测
  2. 调用 `_judge()` 比较原始预测和重定位预测
  3. 根据 Judge 判断选择最优结果
- 返回精炼后的动作字典 + 元数据

#### `predict(task_instruction, obs) -> Tuple`
**主预测方法**（覆盖父类）：
1. 截取截图并编码为 base64
2. 构建多轮对话 messages
3. 调用 UI-TARS API 获取原始预测（最多重试 3 次）
4. 调用 `_maybe_refine_with_bami()` 进行 BAMI 精炼
5. 解析最终动作为 pyautogui 代码
6. 处理 `DONE` / `WAIT` / `FAIL` 特殊动作
7. 返回 `(response_payload, [action_code])`

## BAMI 精炼流程详解

```
原始 UI-TARS 预测
    │
    ▼
┌─ 是否需要 BAMI 精炼？ ──┐
│  (enable_bami=False → 跳过)│
└──────────┬───────────────┘
           │ 是
           ▼
   对每个候选动作：
    │
    ├─ 裁剪截图（mask + expand）
    │  mask_ratio=0.12 遮盖周围区域
    │  crop_expand_ratio=0.2 扩展裁剪
    │
    ├─ 裁剪区域 → UI-TARS 重定位预测
    │  （重新预测目标坐标）
    │
    ├─ Judge 模型比较：
    │   原始预测 vs 重定位预测
    │   └→ 选择 "correct" 的那个
    │
    └─ 输出最终动作 + 精炼元数据
           │
           ▼
    最终 pyautogui 动作
```

## 关键依赖

- `mm_agents.uitars15_v2`：UI-TARS 1.5 基础 Agent 实现
- `PIL`（Pillow）：图像处理、裁剪、绘制
- `vLLM`：本地 Judge 模型推理（可选）
- `OpenAI` / `Doubao API`：远端 VLM 调用

## 与普通 `UITarsAgent` 的差异

| 特性 | `UITarsAgent` | `UITarsBamiAgent` |
|------|--------------|------------------|
| 单次预测 | 1 次 VLM 调用 | 1 次 VLM + 多次本地预测 + 1 次 Judge |
| 坐标精度 | 依赖模型输出 | 通过裁剪 + 重定位修正 |
| 失败容错 | 无重定位 | 原始 vs 重定位选择最优 |
| 速度 | 快 | 较慢（增加 2-4 倍延迟） |
| 适用场景 | 快速评估、成本敏感 | 追求精度、需要细粒度控制 |

## 使用方式

```python
from agent.OSWorld.uitars15_v2_bami import UITarsBamiAgent

agent = UITarsBamiAgent(
    model="doubao-1-5-thinking-vision-pro-250428",
    model_type="doubao",
    enable_bami=True,
    bami_local_judge_model_path="/path/to/judge/model",
    bami_local_judge_gpu="0",
    language="Chinese",
)
```
