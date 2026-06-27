# AgentS3-BAMI

## 1. 环境依赖

### 硬件要求
| 组件 | 配置 | 说明 |
|------|------|------|
| GPU | NVIDIA GPU（24GB+ 显存） | BAMI 和 vLLM 推理加速 |
| 内存 | 32GB+ | 8 个 Docker 容器并行 |
| 存储 | 100GB+ | 模型文件 + 虚拟机镜像 |

### 软件要求
| 软件 | 版本 | 说明 |
|------|------|------|
| Python | 3.10+ | 运行环境 |
| Docker | 20.10+ | OSWorld 容器运行 |
| CUDA | 11.8+ | GPU 加速 |
| NVIDIA Driver | 525+ | GPU 驱动 |

### 外部服务
| 服务 | 端口 | GPU | 说明 |
|------|------|-----|------|
| vLLM Server | 8000 | GPU 6,7 | UI-TARS-1.5-7B（fallback 定位） |
| BAMI Server | 8001 | GPU 4 | Holo2-8B + 裁剪判别模型 |
| Docker | - | CPU | OSWorld 容器（8 并行） |

### Python 依赖

| 依赖包 | 用途 |
| :--- | :--- |
| openai>=1.0.0 | GPT-4o 规划模型调用 |
| pyautogui>=0.9.54 | GUI 鼠标键盘控制 |
| docker>=6.1.0 | OSWorld 容器管理 |
| vllm>=0.4.0 | UI-TARS 推理服务 |
| torch>=2.0.0 | 深度学习框架 |
| pillow>=10.0.0 | 图像处理 |
| pytesseract>=0.3.10 | OCR 文字识别 |
| anthropic>=0.18.0 | Claude 模型支持 |
| backoff>=2.2.0 | API 重试机制 |

### 模型文件

| 模型 | 路径 | 大小 |
| :--- | :--- | :--- |
| UI-TARS-1.5-7B | vLLM 加载 | ~14GB |
| Holo2-8B | /data/models/Holo2-8B | ~16GB |
| 裁剪判别模型 | /data/models/two_box_judge_qwen3_8b/checkpoints/checkpoint-2712 | ~16GB |

### 数据集

| 文件 | 路径 |
| :--- | :--- |
| OSWorld 测试集 | evaluation_examples/test_nogdrive.json |
| Ubuntu 虚拟机镜像 | docker_vm_data/Ubuntu.qcow2 |
| Docker 镜像 | happysixd/osworld-docker |

## 2. 代码仓库

```
s3_bami/
├── __init__.py                         # 包初始化文件
├── agents/                             # Agent 核心实现
│   ├── __init__.py
│   ├── agent_s.py                      # AgentS3 主代理类（规划+执行）
│   ├── worker.py                       # Worker 执行代理（动作生成）
│   ├── grounding.py                    # Grounding 层（BAMI + UI-TARS 调用）
│   └── code_agent.py                   # 代码执行代理（Python/Bash）
├── core/                               # 核心引擎
│   ├── __init__.py
│   ├── engine.py                       # LLM 引擎（OpenAI/Anthropic/vLLM）
│   ├── mllm.py                         # 多模态 Agent 封装
│   └── module.py                       # 基础模块类
├── memory/                             # 记忆模块
│   ├── __init__.py
│   └── procedural_memory.py            # Prompt 模板库（所有系统提示词）
├── bbon/                               # 行为分析模块
│   ├── __init__.py
│   ├── behavior_narrator.py            # 行为叙述器（动作前后对比）
│   └── comparative_judge.py            # 比较评测器（多轨迹对比）
├── utils/                              # 工具函数
│   ├── __init__.py
│   ├── common_utils.py                 # 通用工具（LLM调用、代码解析）
│   ├── formatters.py                   # 响应格式检查器
│   └── local_env.py                    # 本地代码执行环境
└── cli_app.py                          # 命令行入口（交互式/单任务）
```

## 3.  如何运行

### 1. 启动 Grounding 服务

#### 启动 vLLM Server（GPU 6,7）- UI-TARS fallback
CUDA_VISIBLE_DEVICES=6,7 vllm serve UI-TARS-1.5-7B \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name ByteDance-Seed/UI-TARS-1.5-7B \
  --max-model-len 32768 \
  --tensor-parallel-size 2
  
#### 启动 BAMI Server（GPU 4）- 主要定位服务
CUDA_VISIBLE_DEVICES=4 python scripts/bami_grounding_server.py \
  --holo2_model_path /data/models/Holo2-8B \
  --judge_model_path /data/models/two_box_judge_qwen3_8b/checkpoints/checkpoint-2712 \
  --port 8001
  
### 2. 运行 AgentS3-BAMI

#### 交互式模式
python cli_app.py \
  --provider openai \
  --model gpt-4o \
  --ground_provider openai \
  --ground_url "http://localhost:8000/v1" \
  --ground_api_key EMPTY \
  --ground_model ByteDance-Seed/UI-TARS-1.5-7B \
  --grounding_width 1920 \
  --grounding_height 1080
  
运行后输入任务，例如：
text
Query: Open Chrome and search for AgentS3

#### 单任务模式
python cli_app.py \
  --provider openai \
  --model gpt-4o \
  --model_url "https://www.dmxapi.cn/v1" \
  --model_api_key "your-api-key" \
  --ground_provider openai \
  --ground_url "http://localhost:8000/v1" \
  --ground_api_key EMPTY \
  --ground_model ByteDance-Seed/UI-TARS-1.5-7B \
  --grounding_width 1920 \
  --grounding_height 1080 \
  --task "Open Chrome and search for AgentS3"
  
#### 批量测试（使用监控脚本）
##### 后台运行完整监控（推荐）
nohup bash scripts/bash/monitor_agents3_bami.sh > logs/monitor_agents3_bami.log 2>&1 &

#### 查看状态
tail -f logs/monitor_agents3_bami.log

### 3. 配置文件方式

也可以通过 `args.json` 配置文件运行：

```bash
python cli_app.py --config args.json
```

`args.json` 配置示例：

```json
{
  "path_to_vm": "docker_vm_data/Ubuntu.qcow2",
  "provider_name": "docker",
  "headless": true,
  "action_space": "pyautogui",
  "observation_type": "screenshot",
  "num_envs": 8,
  "screen_width": 1920,
  "screen_height": 1080,
  "sleep_after_execution": 3.0,
  "max_steps": 50,
  "max_trajectory_length": 8,
  "domain": "all",
  "test_all_meta_path": "evaluation_examples/test_nogdrive.json",
  "test_config_base_dir": "evaluation_examples",
  "result_dir": "results_agents3_bami",
  "model_provider": "openai",
  "model": "gpt-4o",
  "model_url": "https://www.dmxapi.cn/v1",
  "model_api_key": "sk-YGmlJCXmOcpOiHcYQCL91RwuHI6MYYvTXk8XevZaYBAFWERh",
  "model_temperature": null,
  "ground_provider": "openai",
  "ground_url": "http://localhost:8000/v1",
  "ground_api_key": "EMPTY",
  "ground_model": "ByteDance-Seed/UI-TARS-1.5-7B",
  "grounding_width": 1920,
  "grounding_height": 1080,
  "bami_server_url": "http://127.0.0.1:8001"
}
```


