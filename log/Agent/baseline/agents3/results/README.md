# AgentS3 on OSWorld 复现实验报告

**实验目录：** `results_agents3`  
**实验时间：** 2026-05-21 ～ 2026-05-28  
**模型：** GPT-4o（规划）+ UI-TARS-1.5-7B（视觉定位，本地 vLLM）  
**动作空间：** pyautogui（截图 → GPT-4o 规划 → UI-TARS 坐标定位 → 执行）  
**最大步数：** 50  
**并行环境数：** 8 个 Docker 容器并行  
**测试集：** test_nogdrive.json，共 **361 个任务**（排除需要 Google Drive 的任务）

---

## 一、总体结果

| 指标 | 数值 |
|------|------|
| 评测任务数 | 361 / 361 |
| 完全通过任务数 | 76 |
| **总成功率** | **20.9%**（75.51 / 361）|
| 通过任务平均步数 | 10.2 步 |
| 失败任务平均步数 | 18.1 步 |
| 达到步数上限（≥49步）任务数 | 36（10.0%）|

> **注：** 评分来自各任务目录下的 `result.txt`，由 OSWorld 官方评测函数写入。

---

## 二、实验配置

配置文件：[pyautogui/screenshot/gpt-4o/args.json](pyautogui/screenshot/gpt-4o/args.json)

| 参数 | 值 |
|------|----|
| provider_name | docker |
| model（规划） | gpt-4o |
| model_url | https://www.dmxapi.cn/v1 |
| ground_model（定位） | ByteDance-Seed/UI-TARS-1.5-7B |
| ground_url | http://localhost:8000/v1（本地 vLLM）|
| action_space | pyautogui |
| observation_type | screenshot |
| num_envs | 8 |
| max_steps | 50 |
| max_trajectory_length | 8 |
| sleep_after_execution | 3.0s |
| screen_width × height | 1920 × 1080 |

**架构说明：**

```
AgentS3 worker(s)
    ↓ 规划指令               ↓ 截图 + 坐标描述
  gpt-4o API           UI-TARS vllm (port 8000)
  (dmxapi.cn)              (GPU 6, 7)
                                ↓
                     Docker 容器 (Ubuntu VM)
                     happysixd/osworld-docker
```

---

## 三、分应用结果

| 应用 | 通过 / 总数 | 成功率 |
|------|-------------|--------|
| gimp | 11 / 26 | **42.3%** |
| thunderbird | 6 / 15 | **40.0%** |
| os | 9 / 24 | **37.5%** |
| vlc | 6 / 17 | **35.0%** |
| libreoffice_writer | 8 / 23 | **35.0%** |
| vs_code | 7 / 23 | 30.4% |
| chrome | 7 / 46 | 15.2% |
| libreoffice_impress | 5 / 47 | 10.6% |
| libreoffice_calc | 5 / 47 | 10.6% |
| multi_apps | 12 / 93 | 12.5% |

按大类汇总：

| 类别 | 成功率 |
|------|--------|
| 创意工具（GIMP） | 42.3% |
| 通信（Thunderbird） | 40.0% |
| 系统 / OS | 37.5% |
| 媒体（VLC）/ Writer | 35.0% |
| 开发工具（VS Code） | 30.4% |
| Web 浏览（Chrome） | 15.2% |
| Office 办公（Calc + Impress + Writer） | 15.4% |
| 多应用跨平台 | 12.5% |

---

## 四、失败模式分析

### 4.1 失败任务步数分布

| 步数区间 | 失败任务数 | 占比 | 解读 |
|----------|-----------|------|------|
| 1–10 步 | 133 | **46.7%** | 早期失败（操作起点错误或指令理解偏差）|
| 11–20 步 | 73 | 25.6% | 中途卡死 |
| 21–30 步 | 22 | 7.7% | 长期探索无果 |
| 31–40 步 | 21 | 7.4% | 接近极限仍失败 |
| 41–50 步 | 4 | 1.4% | 几乎耗尽步数 |
| 50+ 步 | 36 | **12.6%** | 陷入循环 / 超步限 |

**对比 UI-TARS 单模型：** AgentS3 失败任务平均步数（**18.1 步**）远低于 UI-TARS（31.7 步），说明 GPT-4o 规划器在无法完成时会更快放弃，而不是无限循环。

### 4.2 典型失败模式

**模式一：早期操作失误（46.7%）**

近半数失败发生在前 10 步，Agent 在任务起点即走错方向：

- `Move the image to the right side on Slide 2`（3步失败）：Agent 点击了错误对象，未能选中目标图片
- `Turn the webpage into a PDF file`（2步失败）：Agent 关闭了浏览器而非触发打印/导出

**模式二：死循环 / 步数耗尽（36 例）**

- `Change search results per page to 50`（**92步**）：前 5 步全是 `time.sleep(1.333)` 等待加载，随后在设置页乱点，从未到达目标选项
- `Enable Do Not Track in Chrome`（**79步**）：Agent 反复在设置菜单循环，无法找到正确路径

**模式三：多应用协作失败（multi_apps，87.5% 失败率）**

- `Find daily paper list on Huggingface and record metadata`：需要浏览器检索 → 切换 Writer 记录，跨应用状态衔接失败
- `Configure environment for word embedding tasks`（51步）：安装依赖 → 配置 → 验证，链路过长，中途失败无法恢复

**模式四：深层菜单导航失败**

- `Set decimal separator as comma in LibreOffice Calc`：需要 Tools → Options → Language Settings → Numbers，路径极深
- `Set 'Dim screen when inactive' to off`：系统设置子菜单层级过深，Agent 找不到目标项

---

## 五、典型成功案例

| 应用 | 步数 | 指令摘要 | 成功原因 |
|------|------|----------|----------|
| chrome | 7 | Make Bing the main search engine | 设置路径浅（Settings → Search engine），视觉目标清晰 |
| chrome | 7 | Clear Amazon cookies and browsing data | 路径固定，History → Clear browsing data |
| gimp | 7 | Add a new layer and name it 'Square' | 标准图层操作，菜单路径固定 |
| gimp | 7 | Rotate image horizontally | Image → Transform 标准操作 |
| gimp | 5 | Convert image to CYMK mode | Image → Mode 单步菜单操作 |
| libreoffice_writer | 4 | Center align the heading | 选中标题 → 点居中按钮，2个操作完成 |
| os | 4 | Change permission of all regular files to 644 | `find . -type f -exec chmod 644 {} \;` 单条命令 |
| thunderbird | 7 | Attach AWS bill PDF to email | 步骤固定，Attach → 选文件 |
| vlc | 3 | Auto-adjust brightness and contrast | 单个偏好设置开关 |
| vs_code | 3 | Change VS Code background to photo in Downloads | 设置路径直接，3步完成 |

**规律：** 成功任务集中于两类——① 单条终端命令可解（OS 类）；② 菜单层级 ≤ 2、操作步骤 ≤ 10 的标准 GUI 操作。

---

## 六、各应用深度分析

### GIMP（42.3%）

操作目标视觉显著、菜单路径固定的任务成功率高（图层操作、格式转换、镜像翻转）。失败集中于需要精确参数调整的任务（背景透明化 92 步、亮度调整 7 步失败）——模型不擅长精确拖拽滑块。

### Thunderbird（40.0%）

简单偏好设置和附件操作表现好。失败案例为复杂规则配置（邮件转发规则、引用格式设置），需要在多级对话框中精确填写参数。

### OS（37.5%）

终端命令类任务成功率高（chmod、文件复制等）。失败集中于图形界面系统设置（时区设置、屏幕熄屏设置），需要在 Ubuntu 设置 App 中多层导航。

### Chrome（15.2%）

受双重限制：① 沙箱无外网，所有需要网络访问的任务（查航班、查酒店、浏览商品）全部失败；② 设置页面层级深，Agent 容易在子菜单迷路。纯本地设置任务（搜索引擎、主页、清除缓存）成功率高。

### LibreOffice Calc / Impress（~11%）

任务普遍涉及复杂公式、图表创建、多步格式设置，UI 控件密集，一次点击偏差即导致全流程失败。

### multi_apps（12.5%）

12 个通过任务均为相对独立的单应用操作（ODS 转 CSV、git push、打开指定文件夹等）。失败的 81 个任务几乎都需要在多个 App 之间传递信息或状态，是当前架构的核心短板。

---

## 七、与 UI-TARS 单模型横向对比

| 系统 | 成功率 | 模型 | GIMP | VS Code | multi_apps |
|------|--------|------|------|---------|------------|
| UI-TARS-1.5-7B（repro_1） | 17.23% | 7B 单模型 | 42.3% | **47.8%** | 1.1% |
| **AgentS3（GPT-4o + UI-TARS）** | **20.9%** | GPT-4o + 7B | 42.3% | 30.4% | **12.5%** |

- AgentS3 整体领先约 **3.7%**，但代价是引入 GPT-4o API 调用，成本高出数倍
- GIMP 两者持平（42.3%），说明该类任务主要依赖定位模型（UI-TARS）而非规划器
- VS Code 上 UI-TARS 反而更好（47.8% vs 30.4%），因为 UI-TARS 可直接生成 shell/JSON 命令绕开 GUI
- multi_apps 上 AgentS3 优势明显（12.5% vs 1.1%），GPT-4o 的跨应用规划能力发挥了作用

---

## 八、结论

1. **总成功率 20.9%**，在简单 / 中等难度任务上表现良好，复杂多步任务是主要瓶颈。
2. **近半数失败（46.7%）发生在前 10 步**，说明 grounding 精度和初始规划质量是核心问题，也是 agents3_bami 实验尝试改进的方向。
3. **12.6% 的任务陷入死循环**，Agent 缺乏"识别无效操作并退出循环"的自我纠错能力。
4. **multi_apps 成功率仅 12.5%**，是最大短板；跨应用任务需要专门的状态追踪和协作机制。
5. 与 UI-TARS 单模型相比，AgentS3 的 GPT-4o 规划器在跨应用场景提升显著，但在可直接命令行解决的场景反而不如纯 UI-TARS。
