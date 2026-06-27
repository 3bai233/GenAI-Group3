# UI-TARS + BAMI OSWorld 基准测试结果

**模型：** UI-TARS-1.5 系列（Doubao API）  
**动作空间：** pyautogui（截图 → UI-TARS 预测坐标 → BAMI 精炼 → 执行）  
**Agent：** `UITarsBamiAgent`（`uitars15_v2_bami.py`）  
**运行器：** `run_multienv_uitars_bami.py`  
**总测例数：** 361（OSWorld 标准测试集，排除需 Google Drive 的 8 个任务）

---

## 总体结果

| 指标 | 数值 |
|------|------|
| 总分 | 46 / 361 |
| **总成功率** | **12.7%** |
| 满分 (score=1.0) | 42 |
| 部分成功 (0<score<1) | 4 |
| 失败 (score=0) | 315 |

---

## 目录结构

```
results_uitars_bami_merged/
├── README.md
├── pyautogui/
│   └── screenshot/
│       └── ByteDance-Seed/
│           └── UI-TARS-1.5-7B/
│               ├── chrome/             (46 测例)
│               ├── gimp/               (26 测例)
│               ├── libreoffice_calc/   (47 测例)
│               ├── libreoffice_impress/ (47 测例)
│               ├── libreoffice_writer/ (23 测例)
│               ├── multi_apps/         (93 测例)
│               ├── os/                 (24 测例)
│               ├── thunderbird/        (15 测例)
│               ├── vlc/                (17 测例)
│               └── vs_code/            (23 测例)
└── summary/
    └── results.json                   (361 条记录)
```

每个测例目录中包含 `traj.jsonl`（动作轨迹）、`result.txt`（得分）、`runtime.log`（日志）、`bami_traj.jsonl`（BAMI 精炼轨迹）、`recording.mp4`（录制视频）。

---

## 分应用结果

| 应用 | 测例数 | 成功 |
|------|--------|---------------|
| chrome | 46 | — |
| gimp | 26 | — |
| libreoffice_calc | 47 | — |
| libreoffice_impress | 47 | — |
| libreoffice_writer | 23 | — |
| multi_apps | 93 | — |
| os | 24 | — |
| thunderbird | 15 | — |
| vlc | 17 | — |
| vs_code | 23 | — |
| **合计** | **361** | **46 (12.7%)** |

> 详细分数见 `summary/results.json`。

---

## 运行配置

| 参数 | 值 |
|------|-----|
| Provider | AWS / Docker |
| 动作空间 | pyautogui |
| 观测类型 | screenshot |
| 模型 | doubao-1-5-thinking-vision-pro-250428 |
| 最大步数 | 15 |
| 语言 | Chinese |
| BAMI | 启用 |

---

## 来源实验

本结果来自三次评测实验的汇总：

| 实验 | 测例数 |
|------|--------|
| Rerun | 209 |
| Tuned V1 (Chrome) | 38 |
| Tuned V1  | 150 |
