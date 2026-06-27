# OSWorld 基线实验（agent）

两个 GUI Agent 基线在 OSWorld 上的运行日志与评测结果。对应代码见 [`code/Agent/baseline/`](../../../code/Agent/baseline/)。

| 基线 | 模型 | 总成功率 | 目录 |
|------|------|----------|------|
| UI-TARS | UI-TARS-1.5-7B（单模型端到端，本地 vLLM） | 10.8% (40/369) | [`uitars/`](uitars/) |
| AgentS3 | GPT-4o 规划 + UI-TARS-1.5-7B 定位（双模型） | 19.9% (72/361) | [`agents3/`](agents3/) |

## 目录结构

```
baseline/
├── agents3/
│   ├── logs/      运行日志（agents3-* 主运行日志 + vllm grounding 服务日志）
│   ├── results/   各任务评测结果：result.txt(评分) / traj.jsonl(轨迹) / instruction.txt
│   └── REPORT.md  实验报告
└── uitars/
    ├── logs/      运行日志（benchmark_full-* 主体 + repro1-resume-* 补跑 + vllm 日志）
    ├── results/   各任务评测结果：result.txt / traj.jsonl / instruction.txt / recording.mp4(录屏)
    └── REPORT.md  实验报告
```

## 说明

- 评测分数以各任务目录下 `result.txt` 为准（score=1.0 计为通过），由 OSWorld 官方评测函数写入。
- **未包含 png 逐步截图**（约 5GB，与 recording.mp4 录屏内容重复，已剔除以控制仓库体积）。
- AgentS3 的 `logs/` 已剔除 monitor 每分钟生成的心跳快照，仅保留实际运行主日志。
