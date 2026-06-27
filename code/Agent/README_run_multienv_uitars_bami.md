# `run_multienv_uitars_bami.py` — UI-TARS + BAMI 多环境并行评测运行器

**文件：** `/share/home/group3/agent/OSWorld/run_multienv_uitars_bami.py`

## 概览

该脚本是 **OSWorld 基准测试** 的分布式评测运行器，支持在多个虚拟环境中并行评估 **UI-TARS + BAMI** Agent。它管理 VM 生命周期、任务分发、结果收集，并提供详细的运行日志。

## 架构

```
main()
    │
    ├─ 读取 test_all.json 获取任务清单
    ├─ 构建任务队列 (multiprocessing.Queue)
    │
    └─ Process("EnvProcess-1")
            │
            ├─ 初始化 DesktopEnv (VM / Docker)
            ├─ 初始化 UITarsBamiAgent
            │
            └─ 循环:
                    ├─ 从队列取任务 (domain, example_id)
                    ├─ 加载任务配置 (JSON)
                    ├─ 调用 run_single_example_with_bami_trace()
                    │   ├─ 环境重置
                    │   ├─ Agent.reset()
                    │   ├─ 循环 step:
                    │   │   ├─ Agent.predict(instruction, obs)
                    │   │   ├─ env.step(action)
                    │   │   ├─ 记录截图 / 轨迹
                    │   │   └─ 检查完成
                    │   └─ 记录结果 (lib_results_logger)
                    └─ 记录共享分数
```

## 命令行参数

### 基础参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--path_to_vm` | `None` | VM 镜像路径 |
| `--headless` | `False` | 无头模式运行 |
| `--action_space` | `pyautogui` | 动作空间类型 |
| `--observation_type` | `screenshot` | 观测类型 |
| `--sleep_after_execution` | `3.0` | 动作执行后休眠秒数 |
| `--max_steps` | `15` | 每任务最大步数 |
| `--test_config_base_dir` | `evaluation_examples` | 测试配置根目录 |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `doubao-1-5-thinking-vision-pro-250428` | 模型 ID |
| `--model_type` | `doubao` | 模型类型 (`doubao` / `qwen25` / `qwen25vl`) |
| `--temperature` | `0` | 采样温度 |
| `--top_p` | `None` | Top-p 采样 |
| `--max_tokens` | `3000` | 最大生成 token 数 |
| `--use_thinking` | `False` | 是否启用 Thought 推理字段 |
| `--max_trajectory_length` | `None` | 历史轨迹最大长度 |
| `--max_image_history_length` | `5` | 历史截图保留帧数 |
| `--language` | `Chinese` | 输出语言 |

### BAMI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable_bami` | `False` | 启用 BAMI 动作精炼 |
| `--bami_local_judge_model_path` | `None` | 本地 Judge 模型路径 |
| `--bami_local_judge_base_model_path` | `None` | 本地 Judge 基座模型路径 |
| `--bami_local_judge_gpu` | `None` | Judge 模型 GPU 设备 |
| `--bami_mask_ratio` | `0.12` | BAMI 裁剪 mask 比例 |
| `--bami_crop_expand_ratio` | `0.2` | BAMI 裁剪扩展比例 |

### 任务 & 环境参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--domain` | `all` | 测试领域（如 `chrome`、`all`） |
| `--test_all_meta_path` | `evaluation_examples/test_all.json` | 任务清单路径 |
| `--result_dir` | `./results_uitars_bami` | 结果保存目录 |
| `--num_envs` | `1` | 并行环境数量 |
| `--log_level` | `INFO` | 日志级别 |
| `--region` | `us-east-1` | AWS 区域 |
| `--provider_name` | `aws` | 提供方 (`aws` / `virtualbox` / `vmware` / `docker`) |
| `--client_password` | `''` | VM 客户端密码 |

## 核心函数

### `config() -> argparse.Namespace`
构建并返回所有 CLI 参数的解析器。

### `distribute_tasks(test_all_meta) -> List[Tuple]`
将 `test_all.json` 中的任务扁平化为 `(domain, example_id)` 元组列表。

### `run_single_example_with_bami_trace(agent, env, example, max_steps, instruction, args, example_result_dir, shared_scores)`
**单任务评测核心**（由 `lib_run_single.run_single_example` 实现）：
1. 环境重置 (`env.reset`)
2. Agent 重置 (`agent.reset`)
3. 等待 60s 确保 VM 就绪
4. 循环执行步骤：
   - `agent.predict()` → 获取动作
   - `env.step()` → 执行动作
   - 保存截图 / 轨迹到结果目录
   - 记录奖励分数
5. 记录结果到 `shared_scores` 和 `results.json`

### `run_env_tasks(task_queue, args, shared_scores)`
**工作进程主循环**（由 `multiprocessing.Process` 运行）：
1. 创建 `DesktopEnv` 实例
2. 创建 `UITarsBamiAgent` 实例
3. 从队列获取任务，依次执行
4. 处理各类型异常（基础设施错误、超时等）
5. 根据 `num_envs` 参数启动多个进程

### `signal_handler(signum, frame)`
优雅关闭：接收 `SIGINT` / `SIGTERM` 时安全退出。

### `main()`
入口函数：
1. 加载 `test_all.json`（或指定 domain 的子集）
2. 构建多进程队列和共享分数列表
3. 注册信号处理器
4. 启动工作进程
5. 等待所有进程完成

## 结果目录结构

```
{result_dir}/
├── pyautogui/
│   └── screenshot/
│       └── {model_name}/
│           ├── {domain}/
│           │   └── {example_id}/
│           │       ├── traj.jsonl          # 轨迹记录
│           │       ├── step_1_*.png        # 各步截图
│           │       ├── step_2_*.png
│           │       └── ...
│           └── ...
└── summary/
    └── results.json       # 汇总结果（JSON 数组）
```

## 异常处理

| 异常类型 | 处理方式 |
|----------|----------|
| 截图为 None（容器无响应） | 重试最多 5 次，间隔 15s |
| 基础设施错误（网络、镜像拉取） | 立即终止任务处理 |
| 一般异常 | 记录错误到 `traj.jsonl`，继续处理下一个任务 |
| SIGINT / SIGTERM | 优雅关闭所有环境 |

## 使用示例

```bash
# 基础运行（1 个环境，全部任务）
python run_multienv_uitars_bami.py \
    --provider_name docker \
    --result_dir ./results_uitars_bami \
    --domain all

# 启用 BAMI 精炼（单 domain）
python run_multienv_uitars_bami.py \
    --provider_name docker \
    --domain chrome \
    --enable_bami \
    --bami_local_judge_model_path /path/to/judge \
    --result_dir ./results_uitars_bami_chrome

# AWS 上多环境并行
python run_multienv_uitars_bami.py \
    --provider_name aws \
    --num_envs 4 \
    --region us-east-1 \
    --result_dir ./results_uitars_bami_aws
```

## 依赖文件

| 文件 | 作用 |
|------|------|
| `lib_run_single.py` | 单任务评测循环实现 |
| `lib_results_logger.py` | 线程安全的结果记录 |
| `uitars15_v2_bami.py` | `UITarsBamiAgent` 类 |
| `mm_agents/uitars15_v2.py` | UI-TARS 1.5 基础 Agent |
| `desktop_env/desktop_env.py` | `DesktopEnv` 环境类 |
| `evaluation_examples/test_all.json` | 任务清单 |
