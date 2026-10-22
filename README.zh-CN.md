<div align="center">

# 零样本时序基础模型，真的比线性外推更强吗？

### 一个在三个公开锂电池老化数据集上的可复现基准

[![repro](https://github.com/ilo-love7/battery-soh-tsfm-benchmark/actions/workflows/repro.yml/badge.svg)](https://github.com/ilo-love7/battery-soh-tsfm-benchmark/actions/workflows/repro.yml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
![platform](https://img.shields.io/badge/platform-CPU-lightgrey)
![models](https://img.shields.io/badge/foundation%20models-5-orange)
![cells](https://img.shields.io/badge/evaluated%20cells-61-red)

<p align="center">
  <a href="./README.md">English</a> · <b>简体中文</b>
</p>

</div>

---

**一个面向电池 SOH 预测的可复现基准**：把 Chronos、TimesFM、Moirai、Lag-Llama 当作**不认识电池的陌生人**——只看一段容量衰减历史、**零训练、零微调**，去预测剩余寿命与未来 SOH，并与 naive_last / naive_linear / exp_decay / ARIMA 四个经典基线在**完全相同的协议**下对比，横跨 MIT-Stanford、NASA PCoE、CALCE UMD 三个公开数据集。

> **一句话结论：零样本基础模型不是普遍更强，它是否更强取决于数据有没有"膝点"。**

三个值得先知道的事实：

- **排名会跨库翻转。** 同一个 Chronos-T5，在 MIT 上比线性外推**好 26%**，在 CALCE 上**差 139%**。模型没变，变的是数据。
- **决定胜负的是 `knee_ratio`**——上下文之后衰减加速了多少——而不是参数量。
- **两个标量打赢 60M 权重。** 2 参数校准头 ≈ 单支电池上的全量微调，全程零梯度。

---

## 目录

- [主要结论](#主要结论)
- [快速开始](#快速开始)
- [基准设计](#基准设计)
- [结果](#结果)
- [概率口径：覆盖率几乎只由「区间宽度 / 误差量级」决定](#概率口径覆盖率几乎只由区间宽度--误差量级决定)
- [机制：不是"更懂电池"，而是"敢不敢外推"](#机制不是更懂电池而是敢不敢外推)
- [2 个参数的校准头](#2-个参数的校准头)
- [在线代理：拐点能从上下文估出来，模型选择不能](#在线代理拐点能从上下文估出来模型选择不能)
- [目录结构](#目录结构)
- [复现性](#复现性)
- [局限](#局限)
- [引用与许可](#引用与许可)

---

## 主要结论

| # | 结论 | 关键数字 |
|---|---|---|
| 1 | 零样本基础模型**不是普遍更强**：5 个模型里只有 Chronos-T5 在两个库上稳定领先 | MIT 3.02 vs 线性 4.07（−26%）；NASA 1.52 vs 3.65 |
| 2 | Chronos-Bolt 与线性外推**几乎等价**，差距统计显著但**没有实际意义** | Δ = +0.13 SOH%，p = 0.002 |
| 3 | 优势取决于**数据有没有膝点**，与模型本身无关 | MIT `knee_ratio` 8.88 → 基础模型赢；CALCE 1.65 → 线性外推赢（1.76–2.39×） |
| 4 | 这个排序**不是起点选择的假象**：40/50/60% 三个起点上 CALCE 结论都不变 | 线性 6.45 / 4.62 / 6.00，均为最优 |
| 5 | **few-shot 微调能翻转结论**：19 支电池把 Bolt 从 4.11 拉到 2.66 | −35%，反超所有零样本模型；1 支电池即 −21% |
| 6 | 微调只迁移"**敢外推**"，不迁移"**该多陡**" | CALCE EOL 命中 0% → 88%，但起点已偏低的电池被外推到接近 0（单支 MAE 109.8） |
| 7 | EOL 指标只在 NASA / CALCE 可用，且样本量小到**不足以支撑显著性结论**；NASA 只作方向性证据（排名随起点变化、种子 CV 30.6%，naive_linear 换整段拟合能从 6.11 降到 2.48） | NASA n=4（闸门后 3）、CALCE n=14（闸门后 8） |
| 8 | 结论对起点与种子稳健：三个起点、三个额外种子下排名均无翻转（NASA 除外） | MIT/CALCE 种子 CV ≤ 8.7%；NASA 30.6% |
| 9 | **区间宽度普遍只有误差量级的 4%–15%**，覆盖率因此塌到 9–22%；反过来，`MPIW/MAE ≥ 1.5` 的 6 组覆盖率都在 76–94%，中间**没有过渡样本** | `MPIW/MAE < 1.0` 的 9 组 PICP 全在 8.8–22.2%；`≥ 1.5` 的 6 组全在 76.1–94.4%（ρ=0.83） |
| 10 | 早期版本的 EOL 命中率**会把失败记成满分**：lag-llama 在 MIT 上 100% 命中，其实比真值早 197 个循环就跌破 80% | 过冲 −197 循环、RUL 相对误差 0.71；改用三量后可排序 |
| 11 | **2 个标量的校准头 ≈ 1 支电池上的全量微调**，且 EOL 命中率反超所有微调档位 | MIT 4.110 → 3.256（−20.8%，微调 1 支为 3.241）；EOL 0% → 47% vs 微调 5–37% |
| 12 | 上一条的横截面规律**是相关、不是因果**：只缩放区间半宽，MIT 的 chronos-bolt 拉宽 15 倍覆盖率只到 43%，要 80% 得 113 倍（CRPS +115%）。**「太宽」倒是单标量能修好** | 9/15 组需 38–2000 倍宽度、CRPS 中位涨 465%；lag-llama@MIT 收窄 0.62× 即 80%、CRPS −25% |
| 13 | **换模型的粒度是库、不是 cell**：拐点能只从上下文估出来，但「逐 cell 选模型」的收益天花板只有 1.7% / 0% / 20%，留一库路由 6/6 都没跑赢「两者取优」 | MIT 天花板 2.98 vs 最优单模型 3.03；CALCE 6.04 → 4.82，但留一库只能到 7.96 |
| 14 | **在线估 `w` 三条路全关**：上下文特征回归 19pp 最好但只到量级；conformal 上下文自测可部署口径 38pp 更差；乘事后修正因子后 14.9pp，仍**输给不用自测的库常数 11.4pp**（净收益 −3.5pp） | 118 个 cell：65pp（不校准）→ 38pp（自测）→ 14.9pp（+`k`）vs 11.4pp（常数基线） |

> 全部数字见 [`04_文档报告/结果表.md`](04_文档报告/结果表.md)、
> 区间与 p 值见 [`03_实验结果/显著性检验.md`](03_实验结果/显著性检验.md)、
> 概率口径与 EOL 重设计见 [`04_文档报告/EOL重设计.md`](04_文档报告/EOL重设计.md)、
> 校准头三方对比见 [`04_文档报告/校准头对比.md`](04_文档报告/校准头对比.md)、
> 宽度干预与在线代理见 [`04_文档报告/区间宽度校准.md`](04_文档报告/区间宽度校准.md)
> 与 [`04_文档报告/在线代理.md`](04_文档报告/在线代理.md)；
> 上下文内自测见 [`04_文档报告/上下文自测.md`](04_文档报告/上下文自测.md)。
> 所有走不通的路集中记在 [`04_文档报告/失败记录.md`](04_文档报告/失败记录.md)
> （含「必须改什么前提才能继续」的前提表）。

---

## 这个基准能用来做什么

| 用途 | 入口 |
|---|---|
| **重跑完整基准**（经典基线 + 基础模型 × 三数据集） | `01_源代码/` 下的运行脚本，见上文「快速开始」 |
| **验证某一条结论**（例如「拐点是否解释胜负」） | `01_源代码/ablate_naive_linear.py` 与各 `analyze_*` 脚本 |
| **复用评估协议**（SOH 指标、CRPS / PICP / MPIW、EOL 重设计、显著性与种子方差） | `01_源代码/evaluation/metrics.py`、`01_源代码/data/protocol.py` |
| **接入你自己的模型**（实现适配器接口即可） | `01_源代码/adapters/base.py` 与 `adapters/calibrated.py` |
| **复现已提交的图 / 检查报告链是否腐烂** | `scripts/check_figures.py`（像素级比对，CI 已接入） |
| **看清"能得出/不能得出"哪些结论** | `04_文档报告/`（完整报告）、`03_实验结果/`（表格）、`NOTICE.md` |

典型用途：① 在声称"新 SOH 模型更好"之前当作**公平对照台**；② 电池健康概率预测的**协议模板**；
③ "基础模型值不值得用"的**证据库**（结论：有拐点且能微调时才值得）。

**仓库不提交原始数据集**：数据链接见 `02_数据集链接/`，回归测试可离线运行。
## 快速开始

### 0. 环境

```bash
git clone https://github.com/ilo-love7/battery-soh-tsfm-benchmark && cd battery-soh-tsfm-benchmark

# 核心依赖（跑基线 + 加载数据即可，无需任何模型库）
bash scripts/setup_env.sh          # macOS / Linux / Git-Bash
# 或：scripts\setup_env.bat         # Windows CMD
```

基础模型依赖见 [`requirements-models.txt`](requirements-models.txt)。**两个环境**是刻意的：

- `venv`（Python 3.13）：Chronos / TimesFM / 全部基线 / few-shot 微调；
- `venv-moirai`（Python **3.12**）：Moirai / Lag-Llama —— uni2ts 硬依赖 `numpy~=1.26`，
  没有 cp313 wheel，与 `torch 2.14 + chronos-forecasting` 无法共存于同一 3.13 环境。

CPU 即可，无需 GPU。本文所有数字都是在 CPU（Windows 11 / torch 2.14.1）上跑出来的。

### 1. 数据

代码只**引用**数据路径，不复制、不改写原始文件。把数据放成下面这样即可零配置：

```
data/
├── MIT/          mit_cells.parquet, mit_cycles.parquet, MATR.zip
├── NASA_PCoE/    1. BatteryAgingARC-FY08Q4/*.mat
└── CALCE_UMD/    CS2_*.zip / CX2_*.zip
```

也可以不搬数据，用环境变量指向真实位置：

```bash
export BATTERY_DATA_ROOT=/path/to/datasets   # 总根
export MIT_CELLS=/path/mit_cells.parquet     # 或逐个覆盖
export NASA_MAT_DIR=/path/1.\ BatteryAgingARC-FY08Q4
export CALCE_DIR=/path/CALCE_UMD
```

或复制 [`01_源代码/configs/local_paths.example.py`](01_源代码/configs/local_paths.example.py)
为 `configs/local_paths.py` 直接写死（该文件已被 `.gitignore` 排除）。
各数据集的来源、字段与获取方式见 [`02_数据集链接/`](02_数据集链接/)。

### 2. 权重

```bash
export HF_ENDPOINT=https://hf-mirror.com      # 国内网络；海外可省略
export HF_HUB_DISABLE_XET=1                   # 镜像不支持 xet，不关会中断

hf download amazon/chronos-bolt-small       --local-dir shared/models/chronos-bolt-small
hf download amazon/chronos-t5-small         --local-dir shared/models/chronos-t5-small
hf download google/timesfm-2.5-200m-pytorch --local-dir shared/models/timesfm-2.5-200m-pytorch
hf download Salesforce/moirai-1.0-R-small   --local-dir shared/models/moirai-1.0-R-small
# Lag-Llama 权重：shared/models/lag-llama/lag-llama.ckpt（来源见 requirements-models.txt）
```

`resolve_model_ref()` 会自动优先用本地目录，**运行时不联网**。

### 3. 跑

最快的路径：从已提交产物重建全部表与图（不跑推理，也不需要数据）。

```bash
bash scripts/run_matrix.sh report
```

然后自己复现推理：

```bash
cd 01_源代码

# 一格：MIT 主测试集上跑全部经典基线
../venv/Scripts/python.exe run_benchmark.py --model all-baselines --dataset mit --split test

# 一个基础模型
../venv/Scripts/python.exe run_benchmark.py --model chronos-bolt --dataset mit --split test

# 跨库
../venv/Scripts/python.exe run_benchmark.py --models chronos-bolt,naive_linear --dataset calce
```

一键全矩阵（表 1–表 5 + 图 1–图 6 的原始数据，约 10 小时 CPU）：

```bash
bash scripts/run_matrix.sh all     # 推理部分：mit / nasa / calce / sens / seed
bash scripts/run_matrix.sh extra   # few-shot 微调 + 跨库迁移 + 消融
bash scripts/run_matrix.sh report  # 纯汇总，不重跑推理
```

`extra` 依赖已跑完的 `mit` / `calce` / `nasa` 推理格；`report` 只读 `03_实验结果/`，
是**从干净克隆复现图与汇总产物**的标准入口（已在多次干净克隆上验证逐位一致）。
它也会跑五项无第三方依赖的回归测试：`tests/test_cache_fingerprint.py`（16 项）、
`tests/test_online_proxy.py`（36 项）、`tests/test_conformal_w.py`（20 项）。

`report` 段有 4 个步骤需要原始数据（`calibrate_width` / `online_proxy` /
`diag_context_range` / `ablate_naive_linear`）。没有数据集时会打印
`[跳过]` 并以退出码 0 结束，**不会**用空结果覆盖已提交产物；
推理段不设这个开关，那里缺数据仍然硬失败。
收尾会打印「N 成功 / M 失败」，有任何一步失败整条链返回非零。

输出在 `03_实验结果/<dataset>-<setting>-<models>/`：`per_cell_*.csv`、`summary.json`
（含协议参数、包版本、权重路径、耗时、随机种子）、`predictions_*.npz`。

<details>
<summary>全部单项分析命令</summary>

```bash
# 汇总表 / 显著性 / 机制 / 种子方差
python report.py
python stats_significance.py
python analyze_shrinkage.py
python analyze_seeds.py

# naive_linear 拟合窗口消融（⌊n/2⌋ vs 整段，同时与已发表数字对账）
python ablate_naive_linear.py

# few-shot 曲线（横轴按**实际**支持集大小 1/4/7/19 绘制）
python plot_fewshot.py

# 概率口径（CRPS / PICP / MPIW + EOL 重设计），产物进二级目录 prob/
bash ../scripts/run_prob.sh all
python analyze_eol2.py

# 2 参数校准头：拟合 → 三库评测 → 三方对比
python calibrate_head.py --dataset mit --split train
bash ../scripts/run_cal.sh all
python analyze_calibration.py

# 图（--runs 用分号分隔：结果目录名本身含逗号，所以不能用逗号分隔）
python figures.py --prefix mit_ \
  --runs "mit-zero-chronos-bolt,chronos,timesfm,naive_linear-ctx512;mit-zero-moirai;mit-zero-lag-llama;mit-zero-naive_last,naive_linear,exp_decay-refactor;mit-zero-arima"
python figures.py --prefix nasa_ \
  --runs "nasa-zero-chronos-bolt,chronos,timesfm,naive_linear,naive_last,exp_decay;nasa-zero-moirai;nasa-zero-lag-llama"
python figures.py --prefix calce_ \
  --runs "calce-zero-chronos-bolt,chronos,timesfm,naive_linear,naive_last,exp_decay;calce-zero-moirai;calce-zero-lag-llama"
```

</details>

---

## 基准设计

### 数据（三个库，61 个 cell）

| 数据集 | 体系 | cells | SOH 观测区间 | `knee_ratio` | 用途 |
|---|---|---|---|---|---|
| [MIT-Stanford](02_数据集链接/MIT_MATR-1.md) | LFP / 18650 | 124（41/43/40 官方三划分） | 98.3 → 79.2 | **8.88** | 主战场：强膝点，且**在 EOL 处截断** |
| [NASA PCoE](02_数据集链接/NASA_PCoE.md) | LCO / 18650 | 4 | 83.3 → 72.4 | 0.65 | 小样本、衰减减速 |
| [CALCE UMD](02_数据集链接/CALCE_UMD.md) | LCO / 棱柱 | 14（27 个 zip 按电压窗口筛出全深度老化） | 84.4 → 42.8 | 1.65 | 反向检验：近线性衰减、区间最宽 |

三个库的 `knee_ratio`（未来斜率 / 上下文斜率）相差一个量级，这是本基准最重要的**数据属性**，
也是后面所有"结论反转"的根源。

### 模型

| 类型 | 模型 | 参数 | 上下文上限 | 解码 |
|---|---|---|---|---|
| 基础模型 | Chronos-Bolt-small | 48 M | 2048 | 分位数（确定性） |
| 基础模型 | Chronos-T5-small | 46 M | 512 | 20 样本取中位 |
| 基础模型 | TimesFM 2.5-200M | 200 M | 权重原生 16384，适配器按 **512** 配置 | 点预测 |
| 基础模型 | Moirai-1.0-R-small | 14 M | 512 | 100 样本取中位 |
| 基础模型 | Lag-Llama | **2.4 M** | **32**（原生） | 10 样本取中位 |
| 基线 | naive_last / naive_linear / exp_decay | — | — | 确定性 |
| 基线 | ARIMA | — | — | 确定性 |

> **关于 `naive_linear`**：它只对上下文**最近 ⌊n/2⌋ 个点**做最小二乘（电池序列开头常有容量平台，
> 纳入拟合会把斜率压平）。作者另跑了"整段拟合"变体：MIT 4.066 → 4.528、CALCE 4.623 → 6.275（本文版本更强），
> **但 NASA 6.111 → 2.478（整段更强）**。因此 NASA 只作方向性证据；该消融与讨论见短文初稿 §3.2 脚注。

### 协议（写死，所有模型同口径）

- **切分**：相对起点 50%（敏感性 40/50/60%），用前半段猜后半段；
- **归一**：`SOH = 容量 / median(上下文前 5 点) × 100`，**因果、无泄漏**；
- **推理**：递归滚动，单次 horizon = 64（Chronos 上限 64，硬预测本就不可能），所有模型一致；
- **上下文**：默认各用自身上限（Bolt 2048 / T5 512 / TimesFM 512 / Moirai 512 / Lag-Llama 32）；另跑 `--max-context 512` 做**公平对照**（主表用 512）。
  注意 TimesFM 2.5 的权重声明支持 16384 上下文，是**适配器按 512 配置**的（`adapters/timesfm.py`），这一点与"它原生只有 512"是两回事；
- **评估真值**：Savitzky–Golay 平滑（窗口 11、阶 2），只用于评估参考，不进模型输入；
- **EOL**：SOH 首次 ≤ 80%；**闸门**——真值起点已 ≤ 80% 的 cell 不计入 EOL 分母（否则"未卜先知"）；
- **红线**：依赖缺失必须报错，**禁止静默返回随机数**。

---

## 结果

### MIT-Stanford 主测试集（43 cells，起点 50%，统一 512 上下文）

| 模型 | MAE ↓ | 95% CI | EOL 命中率 | 耗时 |
|---|---|---|---|---|
| **Chronos-T5-small（零样本）** | **3.02** | [2.40, 3.47] | 0% | 319 s |
| naive_linear | 4.07 | [3.76, 4.48] | 0% | 0.1 s |
| Chronos-Bolt-small（零样本） | 4.11 | [3.77, 4.82] | 0% | 14 s |
| exp_decay | 4.53 | [4.24, 5.52] | 0% | 58 s |
| TimesFM 2.5-200M | 4.84 | [4.44, 5.06] | 0% | 31 s |
| ARIMA | 5.15 | [4.89, 5.56] | 0% | 58 s |
| Moirai-1.0-R-small | 5.33 | [4.72, 5.82] | 0% | 144 s |
| naive_last | 5.65 | [5.19, 6.19] | 0% | 0.1 s |
| Lag-Llama（原生 32 上下文） | 7.05 | [6.38, 8.53] | 100%（提前 197 循环） | 502 s |
| **Chronos-Bolt + few-shot（19 支）** | **2.66** | — | 26% | 14 s |

MIT 内部 OOD（batch3，40 cells）：Chronos-Bolt **2.53** vs naive_linear 2.80 —— 基础模型首次反超。

### 跨库零样本（同一批模型，不重新训练）

| 排名 | NASA PCoE（4 cells，EOL 分母 3） | CALCE UMD（14 cells，EOL 分母 8） |
|---|---|---|
| 1 | **Chronos-T5 1.52**（EOL 3/3，误差 6.2 循环） | **naive_linear 4.62**（EOL 8/8） |
| 2 | exp_decay 3.65 | exp_decay 7.35 |
| 3 | Chronos-Bolt 5.75 | Lag-Llama 9.41 |
| 4 | naive_linear 6.11 | Chronos-T5 11.07 |
| 5 | naive_last 6.38 | Chronos-Bolt 11.94 |
| 6 | Moirai 6.58 | TimesFM 12.33 |
| 7 | TimesFM 7.80 | Moirai 12.51 |
| 8 | Lag-Llama 8.72 | naive_last 12.86 |

**同一个 Chronos-T5，在 MIT 上比线性外推好 26%，在 CALCE 上比线性外推差 139%。**
模型没变，变的是数据。

<p align="center">
  <img src="04_文档报告/figures/mit_fig1_trajectories.png" width="48%" />
  <img src="04_文档报告/figures/calce_fig1_trajectories.png" width="48%" />
</p>
<p align="center"><em>左：MIT（强膝点，基础模型敢于下探）；右：CALCE（近线性，基础模型把预测压成水平线）</em></p>

### 统计显著性：哪些差距可以当结论

- MIT 上 Chronos-T5 vs naive_linear：**Δ = −0.77，p < 0.001**（配对 Wilcoxon，n=43）——真实优势；
- MIT 上 Chronos-Bolt vs naive_linear：**Δ = +0.13，p = 0.002**——**显著地更差，但差得没有意义**；
- CALCE 上 4 个基础模型全部显著差于线性外推（p ≤ 0.001），Lag-Llama 例外（p = 0.078, n.s.）；
- NASA n=4：**样本量不足，不做检验**，只报方向与极宽的 bootstrap 区间。

> 这条"显著但无意义"是本基准刻意保留的方法论点：**p 值只能回答"是否可区分"，
> 不能回答"是否值得换"**。因此全文按效应量 + 机制排序，而不是按 p 值排序。

### 多随机种子方差

采样类模型（Chronos-T5 / Moirai / Lag-Llama）在默认种子之外另跑种子 1/2/3，
共 3 库 × 3 模型 × 3 种子。完整表见 [`03_实验结果/种子方差.md`](03_实验结果/种子方差.md)。

| 库 | Chronos-T5 | Moirai | Lag-Llama |
|---|---|---|---|
| MIT test（43 支） | 3.02 / 3.16 / 3.23 / 3.14（CV **2.7%**） | 5.33 / 5.13 / 5.12 / 5.12（2.0%） | 7.05 / 7.15 / 7.01 / 6.99（1.0%） |
| NASA（4 支） | 1.52 / 2.29 / 1.84 / 1.07（**30.6%**） | 6.58 / 5.95 / 6.55 / 7.20（7.7%） | 8.72 / 8.77 / 9.80 / 9.47（5.8%） |
| CALCE（14 支） | 11.07 / 10.65 / 10.18 / 10.66（3.4%） | 12.51 / 12.55 / 12.49 / 12.60（0.4%） | 9.41 / 7.75 / 9.09 / 9.31（**8.7%**） |

> 每格为「默认种子 / seed1 / seed2 / seed3」的中位 MAE，括号内为变异系数 CV。

- **MIT 与 CALCE 的排名在三个种子上无一处翻转**（CV 最大 8.7%，仍小于与下一名的差距）；
- **NASA 的 CV 达 30.6%**（1.07–2.29），只能当方向性证据；
- 种子极差（MIT T5 为 0.20）远小于 bootstrap CI 宽度（1.07）——**再加种子解决不了"只有 43 支电池"这个约束**。

---

## 概率口径：覆盖率几乎只由「区间宽度 / 误差量级」决定

基础模型原生输出预测分布（Chronos-Bolt 直接出分位数、TimesFM 有连续分位数头、
Moirai / Lag-Llama 是采样解码）。本项目把分位数一起落盘，报 **CRPS**、
**PICP@80%**（名义 80% 区间的实际覆盖率）、**MPIW**（平均区间宽度）。
产物在二级目录 `03_实验结果/prob/`，不污染主表。

| 库 | 模型 | CRPS | PICP@80% | MPIW (SOH%) | MPIW / MAE |
|---|---|---|---|---|---|
| MIT | chronos-bolt | 4.123 | 17.2% | 0.279 | 0.068 |
| MIT | chronos | 3.042 | 14.6% | 0.458 | 0.151 |
| MIT | timesfm | 4.857 | 16.7% | 0.217 | 0.045 |
| MIT | moirai | 5.374 | 15.4% | 0.309 | 0.058 |
| MIT | lag-llama | 7.067 | **94.4%** | 36.765 | 5.21 |
| NASA | chronos-bolt | 5.797 | 22.2% | 3.819 | 0.665 |
| NASA | chronos | 2.948 | **81.0%** | 12.904 | 8.47 |
| NASA | timesfm | 8.597 | **76.2%** | 14.582 | 1.87 |
| NASA | moirai | 8.715 | **76.6%** | 21.516 | 3.27 |
| NASA | lag-llama | 12.532 | **86.6%** | 50.950 | 5.84 |
| CALCE | chronos-bolt | 12.627 | 9.3% | 0.724 | 0.057 |
| CALCE | chronos | 11.140 | 8.8% | 1.435 | 0.130 |
| CALCE | timesfm | 12.450 | 10.9% | 0.833 | 0.068 |
| CALCE | moirai | 12.520 | 9.1% | 0.492 | 0.039 |
| CALCE | lag-llama | 8.339 | **76.1%** | 32.146 | 3.42 |

- **覆盖率几乎只由 `MPIW / MAE` 这个比值决定，而且是个近乎二值的开关**：
  15 个「模型 × 库」组合里，比值 < 1.0 的 9 组 PICP 全部落在 **8.8%–22.2%**，
  比值 ≥ 1.5 的 6 组全部落在 **76.1%–94.4%**，中间**一个样本都没有**
  （0.665 与 1.87 之间是空的；Spearman ρ = 0.83）。
  要让名义 80% 的区间真的覆盖 80%，区间宽度至少要跟误差同量级；
  而绝大多数组合只有误差的 **4%–15%**，窄了 1–2 个数量级。
- 但**别把这个规律当因果**。做干预实验（只缩放区间半宽、`q50` 逐位不动）后：
  把 MIT 的 chronos-bolt 拉宽 **15 倍**（把 `MPIW/MAE` 从 0.068 补到约 1.0，与那些天然
  76% 覆盖率的组合同量级）覆盖率只从 17.2% 升到 **43.0%**；要真到 80% 得拉 **113 倍**，
  代价是 CRPS 从 4.12 涨到 8.88（+115%）。9/15 个组合都需要 38–2000 倍的宽度，
  CRPS 中位涨 465%。**结论：横截面规律是相关、不是因果——那些模型不是"区间宽所以覆盖住"，
  而是分布形状本来就更接近真实误差。形状错的分布等比放大只会又宽又错。**
  反过来，"太宽"是可以用 1 个标量收回来的：Lag-Llama 在 MIT 收窄到 0.62× 后
  覆盖率恰好 80%、CRPS 降 25%。详见
  [`04_文档报告/区间宽度校准.md`](04_文档报告/区间宽度校准.md)。
- **CRPS 与 MAE 高度同向（比值中位数 1.00–1.01）**：区间极窄时 CRPS 几乎退化成
  `|y − q50|`，在窄区间里提供不了比 MAE 更多的排序信息；只有宽度到位后
  （NASA 的 chronos 1.91、lag-llama 0.77–1.39）CRPS 才开始明显区别于 MAE。
  更要紧的是：**CRPS 不能当校准目标**——它奖励"更窄"，11/15 个组合的最优 w 都退化到
  `w→0`。只报 CRPS 会系统性掩盖"区间太窄"这个失败模式，所以本项目同时报 PICP 与 MPIW。
  完整表见 [`04_文档报告/区间宽度校准.md`](04_文档报告/区间宽度校准.md)；
  EOL 三量表见 [`04_文档报告/EOL重设计.md`](04_文档报告/EOL重设计.md)。

> ⚠️ **CRPS ≥ |y − q50| 并不普遍成立**，所以"CRPS 小于同口径 MAE"不是 bug 判据。
> 反例：分位数函数近似 U[0,10]（中位数 5），取 y = −100，点质量分布 CRPS = 105，
> 而该分布 CRPS ≈ 103.8 < 105。真正成立的是 `d/dy CRPS = 2F(y) − 1`——
> 固定分布下 CRPS 在中位数处取全局极小（离散分位函数下是一段平台）。
> 判断概率指标可信与否，只能查**分位数链路与点预测链路是否同一条轨迹**：
> `run_benchmark.py` 已加闸门 `metrics.median_deviation()`，当前 18 个组合
> 全部 `max|point − q50| = 0.0e+00`（分位数保序修复与采样链路的对齐，见
> [`04_文档报告/执行记录-2026-10-20.md`](04_文档报告/执行记录-2026-10-20.md) 坑表第 17、18 条）。

**EOL 指标重设计**：早期版本的"命中率"在 MIT 上恒为 0%（数据在 EOL 处截断），在
lag-llama 上恒为 100%——两个常数都不含信息量。补三个量后（`RUL 相对误差` /
`带符号过冲循环数` / `未穿越裕度`）才能看出：lag-llama 的 100% 命中其实是
**比真值早 197 个循环就跌破 80%**（过冲 −197、RUL 相对误差 0.71），
而零样本 Bolt/T5/TimesFM/Moirai 的"0%"分别是**还差 16.1 / 12.7 / 17.2 / 17.8
个 SOH 点**没触到——差距巨大且可排序。

---

## 机制：不是"更懂电池"，而是"敢不敢外推"

对每个 cell 计算两个量：

- `knee_ratio` = 预测段真值斜率 / 上下文真值斜率（**数据侧**，需要多少加速）；
- `pred_ratio` = 预测段降幅 / 真值降幅（**模型侧**，实际外推了多少）。

| 库 | 需要（`knee_ratio`） | Chronos-T5 给出 | 线性外推给出 | 结果 |
|---|---|---|---|---|
| MIT | 8.88 | 0.32（全场最高） | 0.15 | 基础模型赢 |
| CALCE | 1.65 | **0.01**（几乎水平线） | 0.54 | 线性外推赢 |

定义失配度 `mismatch = |log(pred_ratio / knee_ratio)|`，与 MAE 的 Spearman 秩相关：
MIT **+0.98**、NASA +0.93、CALCE +0.64（均不含 Lag-Llama）——
**"外推幅度是否匹配数据的加速需求"几乎单独决定了排名**。

Lag-Llama 是唯一偏离者（加入后 MIT ρ 掉到 +0.38）：它的降幅总量对得上（`pred_ratio` 1.43），
但**路径是错的**——递归 64 步每轮带噪声，曲线剧烈锯齿化，还会提前 197 循环跌破 80%。
这说明"总降幅"是必要条件而非充分条件。

<p align="center">
  <img src="04_文档报告/figures/fig5_shrinkage.png" width="62%" />
</p>

### 跨库迁移：微调学到的是"敢不敢外推"，不是"该多陡"

把 **MIT 微调后的 Chronos-Bolt（k=19）原样搬到 CALCE / NASA**，不做任何再训练：

| 库 | 零样本 Bolt | MIT 微调后 | 变化 |
|---|---|---|---|
| NASA（4 支） | 5.75 | **4.54** | −21% |
| CALCE 全体中位（14 支） | 11.94 | 13.24 | +11%（变差） |
| CALCE 正常起点（9 支） | 12.61 | **8.15** | **−35%** |
| CALCE 起点已 ≤80%（5 支） | 9.59 | 45.36 | +373%（灾难） |

EOL 命中率：CALCE **0% → 88%**，NASA 0% → 33%。

微调真正迁移过去的是"敢往下画"这个先验（`pred_ratio` 0.016 → 敢外推），
但"该画多陡"没有迁移好：起点已经偏低的电池被按 MIT 先验继续外推到接近 0
（单支 MAE 最高 109.8），把整体中位数反而拉高。
**工程判据：部署前先看起点 SOH，起点已明显低于额定健康度时必须重新标定衰减幅度。**

---

## 2 个参数的校准头

如果收益主要来自"敢不敢外推"这一个自由度，那就不必动 190 MB 权重。冻结
Chronos-Bolt，只在输出端学 2 个数：

```
p' = c + s·(p − c) + b        c = 当前上下文最后一个点，逐 chunk 施加
```

`s` = 模型预测的每步变化量打几折，`b` = 每步额外恒定漂移。第三个加性参数在递归下
与 `b` **不可辨识**（都表现为每步漂移），所以这就是 2 个可辨识参数。
`s > 0` 时变换单调，分位数序天然保持，同一组参数可直接用于概率链路。

拟合：MIT train（batch1），起点 0.5、上下文 512、chunk 64，跨 cell 的 MAE_smooth
中位数作目标，两阶段网格搜索（7×7 粗 + 5×5 细，共 74 次前向，**零梯度**）。

| 拟合集 | s | b | 拟合集内中位 MAE | 同集零样本 | 改善 |
|---|---|---|---|---|---|
| 41 支（默认） | **1.65** | **0.00** | 1.954 | 4.368 | −55.3% |
| 19 支（与微调同数据） | 1.60 | 0.00 | 1.435 | 4.016 | −64.3% |

`s ≈ 1.6`、`b = 0`——**单个参数就够了**，且两组不同数据子集给出几乎相同的 `s`。
这正是机制假设的直接印证：模型不是画错了形状，而是不敢把已看到的斜率推到底。

MIT test 上（全部 ctx512 口径）：

| 方案 | 参数/数据预算 | MAE_smooth | 相对零样本 | EOL 命中率 |
|---|---|---|---|---|
| 零样本 Chronos-Bolt | 0 参数 / 0 支 | 4.110 | — | 0% |
| **2 参数校准头（41 支）** | **2 标量 / 41 支 / 无梯度** | **3.256** | **−20.8%** | **47%** |
| 2 参数校准头（19 支，同微调数据） | 2 标量 / 19 支 / 无梯度 | 3.212 | −21.8% | 40% |
| 微调 1 支 | ~60M 权重 / 1 支 / 300 步 | 3.241 | −21.1% | 5% |
| 微调 4 支 | ~60M 权重 / 4 支 / 300 步 | 2.864 | −30.3% | 33% |
| 微调 8 支 | ~60M 权重 / 8 支 / 300 步 | 2.745 | −33.2% | 37% |
| 微调 19 支 | ~60M 权重 / 19 支 / 300 步 | **2.656** | **−35.4%** | 26% |

- **2 个标量 ≈ 1 支电池上的全量微调**（3.256 vs 3.241，差 0.5%），但不需要任何反向传播；
- 拿到 19 支微调收益的 **60%**，参数量差 7 个数量级；
- **EOL 命中率 47%，反超所有微调档位**（5–37%）；
- 概率侧同步改善：CRPS 4.123 → 3.041，PICP 17.2% → 24.7%（仍远低于名义 80%）。

**边界（重要）**：把 MIT 上拟合的 `s = 1.65` 直接搬到其他库**不成立**——

| 库 | 零样本 MAE | 校准 MAE | Δ% | PICP |
|---|---|---|---|---|
| MIT | 4.110 | 3.256 | −20.8% | 17.2% → 24.7% |
| NASA | 5.747 | 5.361 | −6.7% | 22.2% → 41.5% |
| CALCE | 12.600 | 12.755 | **+1.2%** | 9.3% → 17.2% |

CALCE 上是**负迁移**：`s` 学的是"**这个数据分布**该放大多少"，不是"这个模型的属性"，
必须在目标分布上重新拟合；在线估计 `s` 的三条路都不可行（见下文"在线代理"一节）。

复现：

```bash
# 1) 拟合 2 个参数（约 11 分钟，零梯度）
python 01_源代码/calibrate_head.py --dataset mit --split train

# 2) 在三个库上评测（产物进主表 + prob/ 二级目录）
bash scripts/run_cal.sh all

# 3) 生成三方对比表
python 01_源代码/analyze_calibration.py
```

完整表见 [`04_文档报告/校准头对比.md`](04_文档报告/校准头对比.md)。

---

## 在线代理：拐点能从上下文估出来，模型选择不能

用 9 个纯上下文特征（上下文累计衰减、全段/近段斜率外推降幅、曲率、线性拟合 R²、
残差噪声、二阶差分自相关、上下文长度），在 61 个 cell 上做 Spearman + **留一库**：

| 问题 | 能否只从上下文估 | 证据 |
|---|---|---|
| 拐点 `knee_ratio` | **能** | `drop_ctx` 合并 ρ = **−0.96**，MIT 库内 ρ = **−0.99** |
| 该库是「线性」还是「有拐点」 | **能** | 三库 `drop_ctx` 中位 1.4% / 16.8% / 15.7%，`knee_ratio` 中位 8.9 / 0.65 / 1.65 |
| 该信谁（逐 cell 选模型） | **不能** | 库内 ρ 最高 0.61；留一库路由 **6/6** 没跑赢「两者取优」 |

结论有两条。**逐 cell 选优的天花板本身就很小**——MIT 最优单模型 3.03，6 候选完美选优只有
2.98（**−1.7%**）；NASA 是 0%；只有 CALCE 有 20% 空间（6.04 → 4.82）。而留一库学到的规则
在 CALCE 上只能做到 7.96，还不如直接线性外推的 6.04：模型优劣是**库级**现象，不是 cell 级现象，
「按上下文曲率路由 / 加权集成」没有可赚的钱。可部署的落点是：拿到一个新库，先算上下文的
`drop_ctx` 与 `slope_all`——「已经掉了 15% 以上、后面基本线性」的库（CALCE / NASA 型）
用线性外推，「只掉了 1–2%、拐点还在前方」的库（MIT 型）用基础模型。

### 区间宽度也不能在线估

逐 cell 的区间宽度 `w` 无法从上下文估出。试过两条路：9 个纯上下文特征回归，最强特征其实是
horizon 长度（13/15），覆盖率仍偏 10–20pp；conformal 式上下文内自测（切两段、前段当输入、
后段当真值，在伪任务上求 `w_hat`），118 cell 上覆盖率偏离 65.1pp → 38.2pp，仍显著过度自信。
乘一个事后因子 `k` 能降到 14.9pp，但同样作弊预算下的库常数基线是 11.4pp——净收益为负。
自测还会主动破坏本来就好的模型：NASA 的 timesfm 不校准时 PICP 已是 76.2%，自测把它
「修正」到 48.8%。

```bash
python 01_源代码/online_proxy.py     # 零推理，只读已落盘的 predictions / quantiles
python 01_源代码/conformal_w.py --skip-inference   # 读已提交产物重建文档
```

完整表见 [`04_文档报告/在线代理.md`](04_文档报告/在线代理.md)、
[`04_文档报告/上下文自测.md`](04_文档报告/上下文自测.md)。

---

## 目录结构

```
01_源代码/
├── configs/config.py        路径解析 + 协议参数 + 模型注册表 + HF 镜像（唯一真源）
├── data/loaders.py          MIT / NASA / CALCE 真实加载器（CALCE 带来源指纹的 parquet 缓存）
├── data/protocol.py         清洗 / SOH 归一 / 相对起点切分 / 平滑参考
├── adapters/base.py         Forecaster 接口 + 批量滚动递归 + 分位数接口 + 分位数穿越修复
├── adapters/                chronos / timesfm / lagllama / moirai（各自独立种子推进）
├── adapters/calibrated.py   2 参数单调校准头（冻结基础模型，p' = c + s(p-c) + b）
├── baselines/classic.py     naive_last / naive_linear / exp_decay / ARIMA
├── evaluation/metrics.py    MAE / RMSE(SOH) + EOL 误差 + RUL 误差 + 命中率
│                            + CRPS / PICP / MPIW + EOL 重设计三量
├── tests/                   无第三方依赖的回归测试（缓存指纹 / 在线代理 / 伪任务切分，共 214 项）
├── run_benchmark.py         一行命令复现一格（--models / --max-context / --start-fracs / --seed）
├── calibrate_head.py        拟合校准头的 (s, b)（两阶段网格，零梯度）
├── report.py                汇总 summary.json → 结果表.md + 汇总表.csv
├── stats_significance.py    bootstrap CI + 配对 Wilcoxon → 显著性检验.md
├── analyze_shrinkage.py     pred_ratio / 失配度 / 秩相关 → 衰减保留率.csv + fig5
├── analyze_eol2.py          EOL 重设计汇总 → 汇总_EOL重设计.csv + EOL重设计.md
├── analyze_calibration.py   零样本 / 2 参数 / 微调三方对比 → 校准头对比.md
├── calibrate_width.py       区间宽度干预（绕中位数缩放半宽，零模型重跑）→ 区间宽度校准.md
├── online_proxy.py          在线代理：9 个上下文特征 → 拐点 / 模型选择 / 宽度（零推理）
├── diag_context_range.py    上下文长度可行性核查：长度分布 + 夹到 512 的口径影响（零推理）
├── conformal_w.py           上下文内自测（conformal 式伪任务）→ 上下文自测.md
├── analyze_seeds.py         多种子方差与排名翻转检查 → 种子方差.md
├── diag_curvature.py        knee_ratio（数据侧衰减加速比）
├── ablate_naive_linear.py   naive_linear 拟合窗口消融（⌊n/2⌋ vs 整段）
├── fewshot_finetune.py      Chronos-Bolt 真梯度微调（k 支电池）+ 跨库迁移
├── plot_fewshot.py          fig6：MAE / EOL 随**实际**支持集大小的变化
├── figures.py               轨迹图 / 箱线图 / EOL 命中率 / 起点敏感性
└── 一次性诊断（不进 report 链）
    ├── backfill_ctx.py      给历史 predictions_*.npz 回填 `key|ctx`（免重跑推理）
    ├── diag_calce_files.py  逐个 zip 文件判断"是不是容量老化测试"
    ├── diag_calce_depth.py  按 Vmax/Vmin 判"真正全深度"的循环数（27 → 14 的依据）
    └── diag_calce_cells.py  按放电量/额定容量判全深度（第二条独立依据）
02_数据集链接/                三个数据集的来源、字段与获取方式（只写路径，不存数据）
03_实验结果/                  每次运行的 per_cell / summary / predictions + 汇总与检验
  ├── prob/                  概率口径矩阵（CRPS/PICP/MPIW + EOL 重设计），二级目录不进主表
  ├── calib/                 校准头参数与网格曲面（calibration_head.json / grid_surface.csv）
  ├── width_calib/           区间宽度扫描（width_calibration.csv + 逐组合曲线 + 跨库迁移）
  ├── online_proxy/          逐 cell 上下文特征 / 逐 cell w_cov / Spearman 表 / 路由 / 留一库
  └── conformal_w/           上下文内自测（horizon 不变性 + 逐 cell w_hat / w_oracle / 三种 PICP）
04_文档报告/                  短文初稿、方案、执行记录、结果表、figures、_history（历史材料）
scripts/                      setup_env.{sh,bat} + run_matrix.sh + run_prob.sh + run_cal.sh
                             + check_docs.py（文档链接与锚点）+ check_figures.py（图一致性）
.github/workflows/repro.yml    CI：干净克隆上跑测试 + 汇总链，断言图与产物未被改动
requirements-core.txt         核心依赖（基线所需）
requirements-models.txt       模型依赖（已实测可共存的版本组合 + 踩坑记录）
```

---

## 复现性

所有提交的数字与图都由管道产出，没有手工改数：

```bash
bash scripts/run_matrix.sh report    # 22 步：汇总 + 显著性 + 机制 + 种子方差 + 消融 + 出图
```

- **CI 每次 push 都在干净克隆上重跑**（[`.github/workflows/repro.yml`](.github/workflows/repro.yml)）：
  214 项回归测试 + 22 步汇总链 + 13 个文档的链接与锚点检查，然后断言图与已提交版本一致、
  文本产物零改动。徽章见页面顶部。
- **没有原始数据也能跑 report**：4 个需要数据集的步骤会打印 `[跳过]` 并退出 0，
  不会用空结果覆盖已提交产物；有数据时它们照常运行。
- **图的一致性分两个口径**：同平台要求字节级一致（`python scripts/check_figures.py`），
  跨平台只要求像素级一致（`--pixels`）——FreeType / zlib 差异会让 PNG 字节必然不同，
  这一点由 CI 首次运行实测确认。
- 复现过程中修掉的 20 个坑（每一个都会让数字悄悄错掉、而且不报错）完整记在
  [`04_文档报告/执行记录-2026-10-20.md`](04_文档报告/执行记录-2026-10-20.md)，此处不展开。

---

## 局限

1. **EOL 指标太脆**：MIT 在 EOL 处截断（数据没有"跌破 80% 之后"的样本），零样本模型"差一点"就到不了，
   命中率恒为 0%；NASA 闸门后只剩 3 支，CALCE 剩 8 支，**都不足以做显著性检验**。
   已补三个替代量（RUL 相对误差 / 带符号过冲 / 未穿越裕度），把常数 0 换成了可排序的数；
   但样本量本身仍不足，EOL 只作辅助指标。
2. **Lag-Llama 的上下文被 32 卡住，而且它是五个模型里规模最小的（实测权重 2.4 M 参数，仅为 Chronos-T5-small 的 1/19）**：
   原生上下文与其他模型的 512 不可比，容量差距也极大。
   已单独标注；但它在 MIT 上仍是最差（7.05），说明"提前 197 循环跌破 80%"不能只归因于上下文长度或参数规模。
3. **只有 LFP 与 LCO 两个体系**：没有 NCM/NCA、没有快充工况、没有温度维度。
4. **few-shot 只做了 Chronos-Bolt**：其余四个模型的微调接口差异大，未覆盖。
5. **全部在 CPU 上跑**：单次全矩阵约 10 小时，因此只做了 3 个额外种子（不是 10 个）。
6. **`knee_ratio` 是事后量**：它需要真值才能算，只能用于**解释**已发生的结果，
   不能作为部署时的先验判据。已补在线代理：拐点能用上下文的 `drop_ctx` 估出来（ρ = −0.96），
   但模型选择与逐 cell 区间宽度都估不出来。
7. **校准头的 `s` 不能跨库迁移**：在 MIT 上拟合的 `s = 1.65` 搬到 CALCE 是负迁移（+1.2%）。
   `s` 学的是"这个数据分布该放大多少"，不是模型属性，必须在目标分布上重新拟合。
   区间宽度倍数 `w` 同样不可迁移，而用上下文回归只能把覆盖率误差从 42pp 压到 19pp；
   conformal 式的上下文内自测更差（可部署口径 38pp），加事后修正因子也赢不过一个常数基线。
8. **概率口径仍远未校准**：即使经过校准头，PICP 也只有 25%（MIT）/ 41%（NASA）/ 17%（CALCE），
   离名义 80% 还很远。本基准只能指出"过度自信"这个方向，没有解决它。
9. **"长上下文"这个因子在本数据上不可测**：183 个「cell×起点」组合里没有一个的上下文
   超过 2048（最长 1340），所以 512/1024/2048 的三档对照退化成「512 vs 原生」。
   要研究上下文长度的影响，需要单支电池有 4000 个以上循环的数据集
   （见 [`04_文档报告/上下文范围.md`](04_文档报告/上下文范围.md)）。
10. **逐 cell 选模型没有空间**：6 个候选的逐 cell 完美选优相对最优单模型只改善
    1.7%（MIT）/ 0%（NASA）/ 20%（CALCE），且留一库学不到可迁移的规则。
    集成/路由类方法的收益上限就在这里。
11. **在线估区间宽度 `w` 在本协议下做不到**：特征回归只能到 19pp（留一库），
    conformal 上下文自测可部署口径反而更差（38pp），加事后修正因子仍输给一个常数基线。
    根因是 50/50 切分让「匹配 horizon 的自测」在数学上不存在，而 `w` 对 horizon 极度敏感
    （horizon 减半 ⇒ `w` 只剩 11.7%）。要解决只能改协议（更晚的起点），
    那会改变所有已发表数字的口径（见 [`04_文档报告/上下文自测.md`](04_文档报告/上下文自测.md)）。

---

## 引用与许可

代码 MIT，见 [`LICENSE`](LICENSE)。**数据与权重不随仓库分发**，分发范围与获取方式见 [`NOTICE.md`](NOTICE.md)。

如果这个基准对你有帮助：

```bibtex
@software{zhicheng_tsfm_battery_2026,
  title  = {Do Zero-Shot Time-Series Foundation Models Beat Simple Baselines
            for Battery SOH Prediction? A Three-Dataset Benchmark},
  author = {ilo-love7},
  year   = {2026},
  note   = {https://github.com/ilo-love7/battery-soh-tsfm-benchmark},
  license = {MIT}
}
```

也可直接使用 [`CITATION.cff`](CITATION.cff)（GitHub 会显示 "Cite this repository"）。

---

*最后更新：2026-10-22 ｜ 短文初稿见 [`04_文档报告/短文初稿-v1.md`](04_文档报告/短文初稿-v1.md) ｜ 完整过程记录见 [`04_文档报告/执行记录-2026-10-20.md`](04_文档报告/执行记录-2026-10-20.md)*
