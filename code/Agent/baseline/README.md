# Agent 基线代码（OSWorld）

两个 GUI Agent 基线在 OSWorld 上的运行代码。日志与评测结果见 [`log/Agent/baseline/`](../../../log/Agent/baseline/)。

| 基线 | 模型 | 总成功率 |
|------|------|----------|
| UI-TARS | UI-TARS-1.5-7B（单模型端到端，本地 vLLM） | 10.8% (40/369) |
| AgentS3 | GPT-4o 规划 + UI-TARS-1.5-7B 定位（双模型） | 19.9% (72/361) |

## 目录结构

```
code/Agent/baseline/
├── uitars/                       UI-TARS 单模型基线
│   ├── uitars_agent.py             Agent 主体（感知→推理→动作，输出相对坐标）
│   ├── prompts.py                  Prompt 模板与动作空间
│   └── run_multienv_uitars.py      多环境并行运行入口
├── agents3/                      AgentS3 双模型基线
│   ├── run_agents3.py              多环境并行运行入口
│   ├── run_agents3.sh              单次运行脚本
│   ├── monitor_agents3.sh          监控/自动重启脚本
│   └── gui_agents_s3/              AgentS3 agent 包（干净基线，非 BAMI）
│       ├── agents/agent_s.py         入口类 AgentS3
│       ├── agents/worker.py          Generator + Reflection
│       ├── agents/grounding.py       OSWorldACI（动作接口 + vllm grounding）
│       ├── agents/code_agent.py      CodeAgent（执行 bash/python）
│       ├── memory/procedural_memory.py  系统 prompt / 动作定义
│       └── core, utils, bbon ...     引擎与工具（bbon 为 Agent-S3 自带 Best-of-N）
└── common/
    └── lib_run_single.py         两基线共用的单任务运行器
```

## 来源

- `uitars/`、`agents3/run_*`、`common/`：取自 OSWorld 仓库（`mm_agents/`、`scripts/`、`lib_run_single.py`）。
- `agents3/gui_agents_s3/`：取自 Agent-S 仓库的 `gui_agents/s3/`（干净 AgentS3 基线包，使用 vllm grounding）。其 BAMI 改造版 `gui_agents/s3_bami/` 仅 `agents/grounding.py` 不同（改用 BAMI grounding server），不属于本基线。

仅收录构成基线实现的关键源码，未包含 OSWorld / Agent-S 完整仓库与依赖。运行环境与命令参见各基线的实验报告（`log/Agent/baseline/*/REPORT.md`）。
