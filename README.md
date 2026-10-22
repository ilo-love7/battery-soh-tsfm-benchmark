<div align="center">

# Do Zero-Shot Time-Series Foundation Models Really Beat Linear Extrapolation?

### A reproducible three-dataset benchmark for battery state-of-health prediction

[![repro](https://github.com/ilo-love7/battery-soh-tsfm-benchmark/actions/workflows/repro.yml/badge.svg)](https://github.com/ilo-love7/battery-soh-tsfm-benchmark/actions/workflows/repro.yml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
![platform](https://img.shields.io/badge/platform-CPU-lightgrey)
![models](https://img.shields.io/badge/foundation%20models-5-orange)
![cells](https://img.shields.io/badge/evaluated%20cells-61-red)

<p align="center">
  <b>English</b> · <a href="./README.zh-CN.md">简体中文</a>
</p>

</div>

---

**A reproducible benchmark for battery SOH forecasting** that treats Chronos, TimesFM, Moirai and Lag-Llama as *strangers who know nothing about batteries*: given only a capacity-fade history, with **zero training and zero fine-tuning**, predict future SOH and remaining life — and compare them against `naive_last` / `naive_linear` / `exp_decay` / ARIMA under one identical protocol, across MIT-Stanford, NASA PCoE and CALCE UMD.

> **Short answer: zero-shot foundation models are not universally better. Whether they win depends on whether the data has a knee.**

Three things worth knowing before you scroll:

- **The ranking flips across datasets.** The same Chronos-T5 is 26% *better* than linear extrapolation on MIT and 139% *worse* on CALCE. The model did not change; the data did.
- **What predicts the winner is `knee_ratio`** — how much the fade accelerates after the context window — not parameter count.
- **Two scalars beat 60M weights.** A 2-parameter calibration head matches full fine-tuning on a single cell, with no backpropagation at all.

---

## Contents

- [Key findings](#key-findings)
- [Quick start](#quick-start)
- [Benchmark design](#benchmark-design)
- [Results](#results)
- [Probabilistic metrics: coverage is set almost entirely by interval width](#probabilistic-metrics-coverage-is-set-almost-entirely-by-interval-width)
- [Mechanism: not "understands batteries", but "dares to extrapolate"](#mechanism-not-understands-batteries-but-dares-to-extrapolate)
- [The 2-parameter calibration head](#the-2-parameter-calibration-head)
- [Online proxies: the knee is estimable, model selection is not](#online-proxies-the-knee-is-estimable-model-selection-is-not)
- [Repository layout](#repository-layout)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Citation and license](#citation-and-license)

---

## Key findings

| # | Finding | Key numbers |
|---|---|---|
| 1 | Zero-shot foundation models are **not universally better**: only Chronos-T5 leads consistently, and only on two of the three datasets | MIT 3.02 vs linear 4.07 (−26%); NASA 1.52 vs 3.65 |
| 2 | Chronos-Bolt and linear extrapolation are **practically equivalent** — the gap is statistically significant but **meaningless** | Δ = +0.13 SOH%, p = 0.002 |
| 3 | The advantage depends on **whether the data has a knee**, not on the model | MIT `knee_ratio` 8.88 → foundation model wins; CALCE 1.65 → linear wins (1.76–2.39×) |
| 4 | This ordering is **not an artifact of the start point**: the CALCE conclusion holds at 40/50/60% starts | linear 6.45 / 4.62 / 6.00, best in all three |
| 5 | **Few-shot fine-tuning flips the conclusion**: 19 cells take Bolt from 4.11 to 2.66 | −35%, beating every zero-shot model; a single cell already gives −21% |
| 6 | Fine-tuning transfers "**dare to extrapolate**", not "**how steep to go**" | CALCE EOL hit 0% → 88%, but cells starting below 80% get extrapolated toward 0 (single-cell MAE 109.8) |
| 7 | EOL is only available on NASA / CALCE, and the sample size is **too small to support significance claims**; NASA is directional evidence only (ranking shifts with start point, seed CV 30.6%; full-segment refit takes `naive_linear` from 6.11 to 2.48) | NASA n=4 (3 after gating), CALCE n=14 (8 after gating) |
| 8 | Conclusions are robust to start point and seed: no ranking flips across three starts and three extra seeds (except NASA) | MIT/CALCE seed CV ≤ 8.7%; NASA 30.6% |
| 9 | **Interval width is typically only 4%–15% of the error magnitude**, so coverage collapses to 9–22%; conversely, all 6 groups with `MPIW/MAE ≥ 1.5` land at 76–94% coverage — with **no samples in between** | 9 groups at `MPIW/MAE < 1.0` → PICP 8.8–22.2%; 6 groups at `≥ 1.5` → 76.1–94.4% (ρ=0.83) |
| 10 | The earlier EOL hit-rate metric **scored failures as perfect**: lag-llama hit 100% on MIT while actually crossing 80% **197 cycles early** | overshoot −197 cycles, RUL relative error 0.71; three replacement metrics make it orderable |
| 11 | **A 2-scalar calibration head ≈ full fine-tuning on one cell**, and its EOL hit rate beats every fine-tuning setting | MIT 4.110 → 3.256 (−20.8%, vs 3.241 for 1-cell fine-tuning); EOL 0% → 47% vs 5–37% |
| 12 | The cross-sectional pattern above is **correlation, not causation**: scaling interval half-width only, MIT's chronos-bolt reaches 43% coverage at 15× and needs 113× for 80% (CRPS +115%). **"Too wide" *is* fixable with one scalar** | 9/15 groups need 38–2000× width, median CRPS +465%; lag-llama@MIT narrows 0.62× to hit 80%, CRPS −25% |
| 13 | **Model choice is a per-dataset phenomenon, not a per-cell one**: the knee is estimable from context alone, but per-cell routing caps out at 1.7% / 0% / 20% gain, and leave-one-dataset-out routing loses 6/6 to "take the better of two" | MIT ceiling 2.98 vs best single model 3.03; CALCE 6.04 → 4.82, but LOOCV only reaches 7.96 |
| 14 | **All three routes to online `w` estimation are closed**: context-feature regression reaches 19pp at best; conformal self-testing is worse at 38pp deployable; after a post-hoc correction factor 14.9pp, still **worse than the no-self-test dataset constant at 11.4pp** (net −3.5pp) | 118 cells: 65pp (uncalibrated) → 38pp (self-test) → 14.9pp (+`k`) vs 11.4pp (constant baseline) |

> All numbers: [`04_文档报告/结果表.md`](04_文档报告/结果表.md).
> Intervals and p-values: [`03_实验结果/显著性检验.md`](03_实验结果/显著性检验.md).
> Probabilistic metrics and the EOL redesign: [`04_文档报告/EOL重设计.md`](04_文档报告/EOL重设计.md).
> Three-way calibration comparison: [`04_文档报告/校准头对比.md`](04_文档报告/校准头对比.md).
> Width intervention and online proxies: [`04_文档报告/区间宽度校准.md`](04_文档报告/区间宽度校准.md)
> and [`04_文档报告/在线代理.md`](04_文档报告/在线代理.md);
> in-context self-testing: [`04_文档报告/上下文自测.md`](04_文档报告/上下文自测.md).
> Every dead end is collected in [`04_文档报告/失败记录.md`](04_文档报告/失败记录.md)
> (including a table of "what premise must change before this can continue").
>
> *Document and directory names in this repository are in Chinese; links point at the real paths.*

---

## Quick start

### 0. Environment

```bash
git clone https://github.com/ilo-love7/battery-soh-tsfm-benchmark && cd battery-soh-tsfm-benchmark

# Core deps only (baselines + data loading, no model libraries)
bash scripts/setup_env.sh          # macOS / Linux / Git-Bash
# or: scripts\setup_env.bat         # Windows CMD
```

Foundation-model dependencies live in [`requirements-models.txt`](requirements-models.txt). **Two environments are intentional:**

- `venv` (Python 3.13): Chronos / TimesFM / all baselines / few-shot fine-tuning;
- `venv-moirai` (Python **3.12**): Moirai / Lag-Llama — `uni2ts` hard-pins `numpy~=1.26`, which has no cp313 wheel and cannot coexist with `torch 2.14 + chronos-forecasting` in one 3.13 environment.

CPU is enough; no GPU required. Every number in this README was produced on CPU (Windows 11 / torch 2.14.1).

### 1. Data

The code only **references** data paths — it never copies or rewrites raw files. Lay the data out like this for zero-config:

```
data/
├── MIT/          mit_cells.parquet, mit_cycles.parquet, MATR.zip
├── NASA_PCoE/    1. BatteryAgingARC-FY08Q4/*.mat
└── CALCE_UMD/    CS2_*.zip / CX2_*.zip
```

Or skip moving anything and point environment variables at the real location:

```bash
export BATTERY_DATA_ROOT=/path/to/datasets   # root
export MIT_CELLS=/path/mit_cells.parquet     # or override individually
export NASA_MAT_DIR=/path/1.\ BatteryAgingARC-FY08Q4
export CALCE_DIR=/path/CALCE_UMD
```

Alternatively copy [`01_源代码/configs/local_paths.example.py`](01_源代码/configs/local_paths.example.py) to `configs/local_paths.py` and hard-code paths there (that file is git-ignored). Dataset provenance, fields and download instructions are in [`02_数据集链接/`](02_数据集链接/).

### 2. Weights

```bash
export HF_ENDPOINT=https://hf-mirror.com      # mainland China; omit elsewhere
export HF_HUB_DISABLE_XET=1                   # mirrors do not support xet; omitting this breaks the download

hf download amazon/chronos-bolt-small       --local-dir shared/models/chronos-bolt-small
hf download amazon/chronos-t5-small         --local-dir shared/models/chronos-t5-small
hf download google/timesfm-2.5-200m-pytorch --local-dir shared/models/timesfm-2.5-200m-pytorch
hf download Salesforce/moirai-1.0-R-small   --local-dir shared/models/moirai-1.0-R-small
# Lag-Llama: shared/models/lag-llama/lag-llama.ckpt (source in requirements-models.txt)
```

`resolve_model_ref()` prefers the local directory automatically — **nothing touches the network at run time**.

### 3. Run

Fastest path: rebuild every table and figure from the committed artifacts (no inference, no data needed).

```bash
bash scripts/run_matrix.sh report
```

Then to reproduce inference yourself:

```bash
cd 01_源代码

# One cell: all classical baselines on the MIT test split
../venv/Scripts/python.exe run_benchmark.py --model all-baselines --dataset mit --split test

# One foundation model
../venv/Scripts/python.exe run_benchmark.py --model chronos-bolt --dataset mit --split test

# Cross-dataset
../venv/Scripts/python.exe run_benchmark.py --models chronos-bolt,naive_linear --dataset calce
```

Full matrix (raw data behind tables 1–5 and figures 1–6, ~10 h on CPU):

```bash
bash scripts/run_matrix.sh all     # inference: mit / nasa / calce / sens / seed
bash scripts/run_matrix.sh extra   # few-shot fine-tuning + cross-dataset transfer + ablation
bash scripts/run_matrix.sh report  # summary chain only, no inference
```

`extra` depends on completed `mit` / `calce` / `nasa` cells; `report` reads only `03_实验结果/` and is **the standard entry point for reproducing figures and summary artifacts from a clean clone** (verified bit-identical across several clean clones). It also runs five dependency-free regression suites (214 items in total), including `tests/test_cache_fingerprint.py` (16 items, locking down the four CALCE cache-fingerprint pitfalls), `tests/test_online_proxy.py` (36 items, locking down the online-proxy feature definitions — especially horizon normalisation of `slope_*`) and `tests/test_conformal_w.py` (20 items, locking down the pseudo-task split and the `T_MIN` fallback).

Four `report` steps need raw data (`calibrate_width` / `online_proxy` / `diag_context_range` / `ablate_naive_linear`). Without datasets they print `[skipped]` and exit 0, **never** overwriting committed artifacts with empty results. The inference stage has no such switch — missing data there is a hard failure. The chain prints `N succeeded / M failed` at the end and returns non-zero if any step failed.

Outputs land in `03_实验结果/<dataset>-<setting>-<models>/`: `per_cell_*.csv`, `summary.json` (protocol params, package versions, weight paths, runtime, seed), `predictions_*.npz`.

<details>
<summary>All individual analysis commands</summary>

```bash
# Summary tables / significance / mechanism / seed variance
python report.py
python stats_significance.py
python analyze_shrinkage.py
python analyze_seeds.py

# naive_linear fitting-window ablation (⌊n/2⌋ vs full segment, reconciled against published numbers)
python ablate_naive_linear.py

# few-shot curves (x-axis uses the ACTUAL support-set size 1/4/7/19)
python plot_fewshot.py

# Probabilistic metrics (CRPS / PICP / MPIW + EOL redesign), output under prob/
bash ../scripts/run_prob.sh all
python analyze_eol2.py

# 2-parameter calibration head: fit → evaluate on three datasets → three-way comparison
python calibrate_head.py --dataset mit --split train
bash ../scripts/run_cal.sh all
python analyze_calibration.py

# Figures (--runs is semicolon-separated: result directory names contain commas)
python figures.py --prefix mit_ \
  --runs "mit-zero-chronos-bolt,chronos,timesfm,naive_linear-ctx512;mit-zero-moirai;mit-zero-lag-llama;mit-zero-naive_last,naive_linear,exp_decay-refactor;mit-zero-arima"
python figures.py --prefix nasa_ \
  --runs "nasa-zero-chronos-bolt,chronos,timesfm,naive_linear,naive_last,exp_decay;nasa-zero-moirai;nasa-zero-lag-llama"
python figures.py --prefix calce_ \
  --runs "calce-zero-chronos-bolt,chronos,timesfm,naive_linear,naive_last,exp_decay;calce-zero-moirai;calce-zero-lag-llama"
```

</details>

---

## Benchmark design

### Data (three datasets, 61 cells)

| Dataset | Chemistry | Cells | SOH range observed | `knee_ratio` | Role |
|---|---|---|---|---|---|
| [MIT-Stanford](02_数据集链接/MIT_MATR-1.md) | LFP / 18650 | 124 (official 41/43/40 split) | 98.3 → 79.2 | **8.88** | Main arena: strong knee, and **truncated at EOL** |
| [NASA PCoE](02_数据集链接/NASA_PCoE.md) | LCO / 18650 | 4 | 83.3 → 72.4 | 0.65 | Small sample, decelerating fade |
| [CALCE UMD](02_数据集链接/CALCE_UMD.md) | LCO / prismatic | 14 (27 zips filtered by voltage window) | 84.4 → 42.8 | 1.65 | Counter-test: near-linear fade, widest range |

The three datasets' `knee_ratio` (future slope / context slope) differ by an order of magnitude. This is the benchmark's most important **data property**, and the root of every "conclusion flip" below.

### Models

| Type | Model | Params | Context cap | Decoding |
|---|---|---|---|---|
| Foundation | Chronos-Bolt-small | 48 M | 2048 | quantile (deterministic) |
| Foundation | Chronos-T5-small | 46 M | 512 | median of 20 samples |
| Foundation | TimesFM 2.5-200M | 200 M | weights native 16384, adapter configured at **512** | point forecast |
| Foundation | Moirai-1.0-R-small | 14 M | 512 | median of 100 samples |
| Foundation | Lag-Llama | **2.4 M** | **32** (native) | median of 10 samples |
| Baseline | naive_last / naive_linear / exp_decay | — | — | deterministic |
| Baseline | ARIMA | — | — | deterministic |

> **On `naive_linear`**: it fits least squares only on the **most recent ⌊n/2⌋ points** of the context (battery series often start with a capacity plateau, and including it flattens the slope). A "full-segment fit" variant was also run: MIT 4.066 → 4.528, CALCE 4.623 → 6.275 (this version is stronger), **but NASA 6.111 → 2.478** (full segment is stronger). NASA is therefore treated as directional evidence only; see footnote §3.2 of the short paper.

### Protocol (fixed, identical for every model)

- **Split**: 50% relative start (sensitivity 40/50/60%); predict the second half from the first;
- **Normalisation**: `SOH = capacity / median(first 5 context points) × 100` — causal, no leakage;
- **Inference**: recursive rolling, horizon = 64 per step (Chronos caps at 64, so hard forecasting is impossible by construction); identical for all models;
- **Context**: by default each model uses its own cap (Bolt 2048 / T5 512 / TimesFM 512 / Moirai 512 / Lag-Llama 32); a `--max-context 512` run provides the **fair comparison** (main tables use 512).
  Note TimesFM 2.5 weights declare 16384 support — the **adapter is configured at 512** (`adapters/timesfm.py`), which is not the same as "it natively only has 512";
- **Ground truth for evaluation**: Savitzky–Golay smoothing (window 11, order 2), evaluation reference only, never fed to models;
- **EOL**: first SOH ≤ 80%; **gating** — cells whose ground-truth start is already ≤ 80% are excluded from the EOL denominator (otherwise "clairvoyance");
- **Red line**: missing dependencies must raise, **never silently return random numbers**.

---

## Results

### MIT-Stanford main test set (43 cells, 50% start, unified 512 context)

| Model | MAE ↓ | 95% CI | EOL hit rate | Time |
|---|---|---|---|---|
| **Chronos-T5-small (zero-shot)** | **3.02** | [2.40, 3.47] | 0% | 319 s |
| naive_linear | 4.07 | [3.76, 4.48] | 0% | 0.1 s |
| Chronos-Bolt-small (zero-shot) | 4.11 | [3.77, 4.82] | 0% | 14 s |
| exp_decay | 4.53 | [4.24, 5.52] | 0% | 58 s |
| TimesFM 2.5-200M | 4.84 | [4.44, 5.06] | 0% | 31 s |
| ARIMA | 5.15 | [4.89, 5.56] | 0% | 58 s |
| Moirai-1.0-R-small | 5.33 | [4.72, 5.82] | 0% | 144 s |
| naive_last | 5.65 | [5.19, 6.19] | 0% | 0.1 s |
| Lag-Llama (native 32 context) | 7.05 | [6.38, 8.53] | 100% (197 cycles early) | 502 s |
| **Chronos-Bolt + few-shot (19 cells)** | **2.66** | — | 26% | 14 s |

MIT internal OOD (batch3, 40 cells): Chronos-Bolt **2.53** vs naive_linear 2.80 — the first time a foundation model wins.

### Cross-dataset zero-shot (same models, no retraining)

| Rank | NASA PCoE (4 cells, EOL denominator 3) | CALCE UMD (14 cells, EOL denominator 8) |
|---|---|---|
| 1 | **Chronos-T5 1.52** (EOL 3/3, error 6.2 cycles) | **naive_linear 4.62** (EOL 8/8) |
| 2 | exp_decay 3.65 | exp_decay 7.35 |
| 3 | Chronos-Bolt 5.75 | Lag-Llama 9.41 |
| 4 | naive_linear 6.11 | Chronos-T5 11.07 |
| 5 | naive_last 6.38 | Chronos-Bolt 11.94 |
| 6 | Moirai 6.58 | TimesFM 12.33 |
| 7 | TimesFM 7.80 | Moirai 12.51 |
| 8 | Lag-Llama 8.72 | naive_last 12.86 |

**The same Chronos-T5 is 26% better than linear extrapolation on MIT and 139% worse on CALCE.** The model did not change; the data did.

<p align="center">
  <img src="04_文档报告/figures/mit_fig1_trajectories.png" width="48%" />
  <img src="04_文档报告/figures/calce_fig1_trajectories.png" width="48%" />
</p>
<p align="center"><em>Left: MIT (strong knee — the foundation model dares to dive); right: CALCE (near-linear — it flattens the forecast)</em></p>

### Statistical significance: which gaps are real conclusions

- MIT, Chronos-T5 vs naive_linear: **Δ = −0.77, p < 0.001** (paired Wilcoxon, n=43) — a genuine advantage;
- MIT, Chronos-Bolt vs naive_linear: **Δ = +0.13, p = 0.002** — **significantly worse, and the difference means nothing**;
- CALCE: all four foundation models significantly worse than linear extrapolation (p ≤ 0.001); Lag-Llama is the exception (p = 0.078, n.s.);
- NASA n=4: **insufficient sample, no test run**; direction and very wide bootstrap intervals reported only.

> This "significant but meaningless" case is a deliberate methodological point: **a p-value answers "are they distinguishable", not "is it worth switching"**. Everything below is ordered by effect size and mechanism, not by p-value.

### Multi-seed variance

Sampling-based models (Chronos-T5 / Moirai / Lag-Llama) were additionally run with seeds 1/2/3: 3 datasets × 3 models × 3 seeds. Full table in [`03_实验结果/种子方差.md`](03_实验结果/种子方差.md).

| Dataset | Chronos-T5 | Moirai | Lag-Llama |
|---|---|---|---|
| MIT test (43 cells) | 3.02 / 3.16 / 3.23 / 3.14 (CV **2.7%**) | 5.33 / 5.13 / 5.12 / 5.12 (2.0%) | 7.05 / 7.15 / 7.01 / 6.99 (1.0%) |
| NASA (4 cells) | 1.52 / 2.29 / 1.84 / 1.07 (**30.6%**) | 6.58 / 5.95 / 6.55 / 7.20 (7.7%) | 8.72 / 8.77 / 9.80 / 9.47 (5.8%) |
| CALCE (14 cells) | 11.07 / 10.65 / 10.18 / 10.66 (3.4%) | 12.51 / 12.55 / 12.49 / 12.60 (0.4%) | 9.41 / 7.75 / 9.09 / 9.31 (**8.7%**) |

> Each cell is median MAE for "default seed / seed1 / seed2 / seed3"; CV in parentheses.

- **No ranking flips on MIT or CALCE across three seeds** (max CV 8.7%, still smaller than the gap to the next rank);
- **NASA's CV reaches 30.6%** (1.07–2.29) — directional evidence only;
- Seed range (MIT T5: 0.20) is far smaller than the bootstrap CI width (1.07) — **more seeds cannot fix "only 43 cells"**.

---

## Probabilistic metrics: coverage is set almost entirely by interval width

Foundation models emit predictive distributions natively (Chronos-Bolt outputs quantiles directly, TimesFM has a continuous quantile head, Moirai / Lag-Llama decode by sampling). Quantiles are persisted alongside point forecasts, and we report **CRPS**, **PICP@80%** (actual coverage of the nominal 80% interval) and **MPIW** (mean prediction interval width). Artifacts live in `03_实验结果/prob/` and do not pollute the main tables.

| Dataset | Model | CRPS | PICP@80% | MPIW (SOH%) | MPIW / MAE |
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

- **Coverage is governed almost entirely by `MPIW / MAE`, and it behaves like a near-binary switch.** Across 15 model × dataset combinations, the 9 groups with ratio < 1.0 all land in **8.8%–22.2%** PICP, and the 6 groups with ratio ≥ 1.5 all land in **76.1%–94.4%** — **not a single sample in between** (the gap between 0.665 and 1.87 is empty; Spearman ρ = 0.83). For a nominal 80% interval to actually cover 80%, its width must be at least on the order of the error; most combinations sit at **4%–15%** of the error, one to two orders of magnitude too narrow.
- **But do not read that as causation.** An intervention that scales only the interval half-width (leaving `q50` untouched) shows: widening MIT's chronos-bolt **15×** (bringing `MPIW/MAE` from 0.068 to about 1.0, on par with the groups that natively hit 76% coverage) raises coverage only from 17.2% to **43.0%**; reaching 80% requires **113×**, at the cost of CRPS rising from 4.12 to 8.88 (+115%). 9 of 15 combinations need 38–2000× width, median CRPS +465%. **The cross-sectional pattern is correlation, not causation — those models do not "cover because they are wide", their distribution shape happens to match the error better. Scaling a misshapen distribution just makes it wide and wrong.** Conversely, "too wide" *is* recoverable with a single scalar: narrowing Lag-Llama on MIT to 0.62× lands exactly at 80% coverage with CRPS down 25%. See [`04_文档报告/区间宽度校准.md`](04_文档报告/区间宽度校准.md).
- **CRPS tracks MAE closely (median ratio 1.00–1.01)**: with very narrow intervals CRPS degenerates to `|y − q50|` and provides no extra ordering information; only once width is adequate (NASA chronos 1.91, lag-llama 0.77–1.39) does CRPS separate from MAE. More importantly, **CRPS must not be used as a calibration target** — it rewards narrowness, and the optimal `w` degenerates to `w→0` in 11/15 combinations. Reporting CRPS alone systematically hides the "intervals are too narrow" failure mode, which is why this benchmark reports PICP and MPIW alongside it.

> ⚠️ **`CRPS ≥ |y − q50|` does not hold in general**, so "CRPS below same-protocol MAE" is not a bug criterion. Counter-example: with a quantile function approximating U[0,10] (median 5), taking y = −100 gives a point-mass CRPS of 105, while the distribution's CRPS ≈ 103.8 < 105. What does hold is `d/dy CRPS = 2F(y) − 1` — for a fixed distribution, CRPS attains its global minimum at the median (a plateau under a discrete quantile function). The only way to judge whether probabilistic metrics are trustworthy is to check whether the quantile path and the point-forecast path are the same trajectory: `run_benchmark.py` now gates on `metrics.median_deviation()`, and all 18 current combinations report `max|point − q50| = 0.0e+00` (see pitfalls 17 and 18 in [`04_文档报告/执行记录-2026-10-20.md`](04_文档报告/执行记录-2026-10-20.md)).

**EOL metric redesign.** The earlier "hit rate" was a constant on both ends: 0% on MIT (data is truncated at EOL) and 100% for lag-llama — neither carries information. Adding three quantities (`RUL relative error` / `signed overshoot in cycles` / `margin to crossing`) makes it orderable: lag-llama's 100% is actually **crossing 80% 197 cycles early** (overshoot −197, RUL relative error 0.71), while the "0%" of zero-shot Bolt/T5/TimesFM/Moirai means they are still **16.1 / 12.7 / 17.2 / 17.8 SOH points short** of crossing — a large and orderable difference.

---

## Mechanism: not "understands batteries", but "dares to extrapolate"

For each cell we compute two quantities:

- `knee_ratio` = ground-truth slope after the split / ground-truth slope in context (**data side**: how much acceleration is needed);
- `pred_ratio` = predicted drop after the split / true drop (**model side**: how much it actually extrapolated).

| Dataset | Needed (`knee_ratio`) | Chronos-T5 delivers | Linear extrapolation delivers | Outcome |
|---|---|---|---|---|
| MIT | 8.88 | 0.32 (highest of all) | 0.15 | foundation model wins |
| CALCE | 1.65 | **0.01** (almost a flat line) | 0.54 | linear extrapolation wins |

Define mismatch `mismatch = |log(pred_ratio / knee_ratio)|`. Its Spearman rank correlation with MAE: MIT **+0.98**, NASA +0.93, CALCE +0.64 (all excluding Lag-Llama) — **whether the extrapolation magnitude matches the data's required acceleration nearly determines the ranking on its own**.

Lag-Llama is the sole outlier (including it drops MIT ρ to +0.38): its total drop matches (`pred_ratio` 1.43) but **the path is wrong** — recursive 64-step decoding adds noise each round, the curve becomes violently saw-toothed, and it crosses 80% 197 cycles early. Total drop is necessary, not sufficient.

<p align="center">
  <img src="04_文档报告/figures/fig5_shrinkage.png" width="62%" />
</p>

### Cross-dataset transfer: fine-tuning learns "dare to extrapolate", not "how steep"

Moving **Chronos-Bolt fine-tuned on MIT (k=19)** to CALCE / NASA unchanged, with no retraining:

| Dataset | Zero-shot Bolt | After MIT fine-tuning | Change |
|---|---|---|---|
| NASA (4 cells) | 5.75 | **4.54** | −21% |
| CALCE all, median (14 cells) | 11.94 | 13.24 | +11% (worse) |
| CALCE normal starts (9 cells) | 12.61 | **8.15** | **−35%** |
| CALCE starts already ≤80% (5 cells) | 9.59 | 45.36 | +373% (disaster) |

EOL hit rate: CALCE **0% → 88%**, NASA 0% → 33%.

What transfers is the prior "dare to draw downward" (`pred_ratio` 0.016 → willing to extrapolate); "how steep to draw" does not transfer well — cells starting low get pushed toward 0 under the MIT prior (single-cell MAE up to 109.8), which drags the overall median up. **Engineering rule: check the starting SOH before deployment; if it is already well below rated health, the fade amplitude must be re-calibrated.**

---

## The 2-parameter calibration head

If the gain comes mainly from one degree of freedom — "dare to extrapolate" — there is no need to touch 190 MB of weights. Freeze Chronos-Bolt and learn just two numbers at the output:

```
p' = c + s·(p − c) + b        c = last point of the current context, applied per chunk
```

`s` = what fraction of the model's per-step change to keep; `b` = a constant per-step drift. A third additive parameter is **unidentifiable** under recursion (both manifest as per-step drift), so these are the 2 identifiable parameters. When `s > 0` the transform is monotone, quantile order is preserved, and the same parameters apply directly to the probabilistic path.

Fitting: MIT train (batch1), 50% start, context 512, chunk 64, targeting the cross-cell median `MAE_smooth`; two-stage grid search (7×7 coarse + 5×5 fine, 74 forward passes, **zero gradients**).

| Fitting set | s | b | Median MAE in-set | Zero-shot same set | Improvement |
|---|---|---|---|---|---|
| 41 cells (default) | **1.65** | **0.00** | 1.954 | 4.368 | −55.3% |
| 19 cells (same data as fine-tuning) | 1.60 | 0.00 | 1.435 | 4.016 | −64.3% |

`s ≈ 1.6`, `b = 0` — **a single parameter suffices**, and two different data subsets yield nearly identical `s`. That is direct confirmation of the mechanism hypothesis: the model is not drawing the wrong shape, it just will not carry the observed slope to its conclusion.

On the MIT test set (all at ctx512):

| Method | Params / data budget | MAE_smooth | vs zero-shot | EOL hit rate |
|---|---|---|---|---|
| Zero-shot Chronos-Bolt | 0 params / 0 cells | 4.110 | — | 0% |
| **2-parameter head (41 cells)** | **2 scalars / 41 cells / no gradients** | **3.256** | **−20.8%** | **47%** |
| 2-parameter head (19 cells, same as fine-tuning) | 2 scalars / 19 cells / no gradients | 3.212 | −21.8% | 40% |
| Fine-tune 1 cell | ~60M weights / 1 cell / 300 steps | 3.241 | −21.1% | 5% |
| Fine-tune 4 cells | ~60M weights / 4 cells / 300 steps | 2.864 | −30.3% | 33% |
| Fine-tune 8 cells | ~60M weights / 8 cells / 300 steps | 2.745 | −33.2% | 37% |
| Fine-tune 19 cells | ~60M weights / 19 cells / 300 steps | **2.656** | **−35.4%** | 26% |

- **2 scalars ≈ full fine-tuning on 1 cell** (3.256 vs 3.241, a 0.5% gap), with no backpropagation;
- Captures **60%** of the 19-cell fine-tuning gain at 7 orders of magnitude fewer parameters;
- **EOL hit rate 47%, beating every fine-tuning setting** (5–37%);
- Probabilistic side improves too: CRPS 4.123 → 3.041, PICP 17.2% → 24.7% (still far below nominal 80%).

**Boundary (important)**: carrying MIT's fitted `s = 1.65` to other datasets **does not work** —

| Dataset | Zero-shot MAE | Calibrated MAE | Δ% | PICP |
|---|---|---|---|---|
| MIT | 4.110 | 3.256 | −20.8% | 17.2% → 24.7% |
| NASA | 5.747 | 5.361 | −6.7% | 22.2% → 41.5% |
| CALCE | 12.600 | 12.755 | **+1.2%** | 9.3% → 17.2% |

CALCE shows **negative transfer**: `s` learns "how much to amplify *this data distribution*", not a property of the model, so it must be refitted on the target distribution. All three routes to estimating `s` online fail (see the online-proxy section).

```bash
# 1) Fit the 2 parameters (~11 min, zero gradients)
python 01_源代码/calibrate_head.py --dataset mit --split train

# 2) Evaluate on all three datasets (artifacts go to main tables + prob/)
bash scripts/run_cal.sh all

# 3) Build the three-way comparison table
python 01_源代码/analyze_calibration.py
```

Full table: [`04_文档报告/校准头对比.md`](04_文档报告/校准头对比.md).

---

## Online proxies: the knee is estimable, model selection is not

Using 9 pure context features (cumulative context drop, full/near-segment slope-extrapolated drop, curvature, linear-fit R², residual noise, second-difference autocorrelation, context length) across 61 cells, with Spearman plus **leave-one-dataset-out**:

| Question | Estimable from context alone? | Evidence |
|---|---|---|
| Knee `knee_ratio` | **Yes** | `drop_ctx` pooled ρ = **−0.96**, within-MIT ρ = **−0.99** |
| Is this dataset "linear" or "kneed" | **Yes** | three-dataset `drop_ctx` median 1.4% / 16.8% / 15.7%; `knee_ratio` median 8.9 / 0.65 / 1.65 |
| Who to trust (per-cell model selection) | **No** | within-dataset ρ max 0.61; leave-one-dataset-out routing loses **6/6** to "take the better of two" |

Two conclusions. **The ceiling for per-cell selection is small to begin with** — MIT best single model 3.03, perfect 6-candidate oracle only 2.98 (**−1.7%**); NASA 0%; only CALCE has 20% headroom (6.04 → 4.82). And the rule learned by leave-one-dataset-out reaches only 7.96 on CALCE, worse than plain linear extrapolation at 6.04: **model superiority is a dataset-level phenomenon, not a cell-level one**, so routing or weighted ensembling by context curvature has no money in it. The deployable takeaway: for a new dataset, compute `drop_ctx` and `slope_all` on the context — "already dropped 15%+, then basically linear" datasets (CALCE / NASA type) call for linear extrapolation, "dropped only 1–2%, knee still ahead" datasets (MIT type) call for foundation models.

### Interval width cannot be estimated online either

Per-cell interval width `w` cannot be recovered from context. Two routes were tried: regression on 9 pure context features (the strongest feature is actually horizon length, 13/15), still 10–20pp off on coverage; and conformal-style in-context self-testing (split in two, first half as input, second half as truth, solve `w_hat` on a pseudo-task), which moves coverage deviation from 65.1pp to 38.2pp across 118 cells — still significantly overconfident. A post-hoc factor `k` brings it to 14.9pp, but a dataset-constant baseline under the same cheating budget reaches 11.4pp — **net negative**. Self-testing also actively damages models that were already fine: NASA's timesfm is at 76.2% PICP uncalibrated, and self-testing "corrects" it down to 48.8%.

```bash
python 01_源代码/online_proxy.py     # zero inference, reads persisted predictions / quantiles
python 01_源代码/conformal_w.py --skip-inference   # rebuild docs from committed artifacts
```

Full tables: [`04_文档报告/在线代理.md`](04_文档报告/在线代理.md), [`04_文档报告/上下文自测.md`](04_文档报告/上下文自测.md).

---

## Repository layout

```
01_源代码/
├── configs/config.py         path resolution + protocol params + model registry + HF mirror (single source of truth)
├── data/loaders.py           real MIT / NASA / CALCE loaders (CALCE uses a provenance-fingerprinted parquet cache)
├── data/protocol.py          cleaning / SOH normalisation / relative-start split / smoothed reference
├── adapters/base.py          Forecaster interface + batched rolling recursion + quantile interface + crossing repair
├── adapters/                 chronos / timesfm / lagllama / moirai (independent seed advancement each)
├── adapters/calibrated.py    2-parameter monotone calibration head (frozen backbone, p' = c + s(p-c) + b)
├── baselines/classic.py      naive_last / naive_linear / exp_decay / ARIMA
├── evaluation/metrics.py     MAE / RMSE(SOH) + EOL error + RUL error + hit rate
│                             + CRPS / PICP / MPIW + the three redesigned EOL quantities
├── tests/                    dependency-free regression suites (cache fingerprint / online proxy / pseudo-task split, 214 items)
├── run_benchmark.py          one command per cell (--models / --max-context / --start-fracs / --seed)
├── calibrate_head.py         fit the calibration head (s, b) (two-stage grid, zero gradients)
├── report.py                 summary.json → 结果表.md + 汇总表.csv
├── stats_significance.py     bootstrap CI + paired Wilcoxon → 显著性检验.md
├── analyze_shrinkage.py      pred_ratio / mismatch / rank correlation → 衰减保留率.csv + fig5
├── analyze_eol2.py           redesigned EOL summary → 汇总_EOL重设计.csv + EOL重设计.md
├── analyze_calibration.py    zero-shot / 2-parameter / fine-tuned three-way comparison → 校准头对比.md
├── calibrate_width.py        interval width intervention (scale half-width around median, no model rerun) → 区间宽度校准.md
├── online_proxy.py           online proxies: 9 context features → knee / model choice / width (zero inference)
├── diag_context_range.py     context-length feasibility: length distribution + effect of clamping to 512 (zero inference)
├── conformal_w.py            in-context self-testing (conformal pseudo-task) → 上下文自测.md
├── analyze_seeds.py          multi-seed variance and ranking-flip checks → 种子方差.md
├── diag_curvature.py         knee_ratio (data-side fade acceleration ratio)
├── ablate_naive_linear.py    naive_linear fitting-window ablation (⌊n/2⌋ vs full segment)
├── fewshot_finetune.py       real-gradient fine-tuning of Chronos-Bolt (k cells) + cross-dataset transfer
├── plot_fewshot.py           fig6: MAE / EOL vs ACTUAL support-set size
├── figures.py                trajectory plots / box plots / EOL hit rate / start-point sensitivity
└── one-off diagnostics (not in the report chain)
    ├── backfill_ctx.py       backfill `key|ctx` into historical predictions_*.npz (no inference rerun)
    ├── diag_calce_files.py   per-zip check: "is this a capacity aging test"
    ├── diag_calce_depth.py   cycles at true full depth by Vmax/Vmin (the basis for 27 → 14)
    └── diag_calce_cells.py   full-depth by discharge/rated capacity (second independent basis)
02_数据集链接/                 provenance, fields and access for the three datasets (paths only, no data)
03_实验结果/                   per-cell / summary / predictions per run + aggregates and tests
  ├── prob/                   probabilistic metric matrix (CRPS/PICP/MPIW + EOL redesign), not in main tables
  ├── calib/                  calibration-head params and grid surface
  ├── width_calib/            interval width sweep + per-combination curves + cross-dataset transfer
  ├── online_proxy/           per-cell context features / per-cell w_cov / Spearman tables / routing / LOOCV
  └── conformal_w/            in-context self-testing (horizon invariance + per-cell w_hat / w_oracle / three PICP variants)
04_文档报告/                   short paper draft, plan, execution log, result tables, figures, _history
scripts/                      setup_env.{sh,bat} + run_matrix.sh + run_prob.sh + run_cal.sh
                              + check_docs.py (doc links and anchors) + check_figures.py (figure consistency)
.github/workflows/repro.yml   CI: tests + summary chain on a clean clone, asserts figures and artifacts unchanged
requirements-core.txt         core dependencies (baselines)
requirements-models.txt       model dependencies (tested-coexistent version sets + pitfall notes)
```

---

## Reproducibility

Every committed number and figure is produced by the pipeline — nothing is hand-edited:

```bash
bash scripts/run_matrix.sh report    # 22 steps: summaries + significance + mechanism + seeds + ablation + figures
```

- **CI reruns on a clean clone on every push** ([`.github/workflows/repro.yml`](.github/workflows/repro.yml)): 214 regression tests + a 22-step summary chain + link and anchor checks across 13 documents, then asserts figures match the committed versions and text artifacts are unchanged byte-for-byte.
- **`report` runs without raw data**: the 4 data-dependent steps print `[skipped]` and exit 0, never overwriting committed artifacts; with data present they run normally.
- **Figure consistency has two modes**: byte-identical on the same platform (`python scripts/check_figures.py`), pixel-identical across platforms (`--pixels`) — FreeType / zlib differences make byte-identical PNGs impossible, confirmed empirically on the first CI run.
- The 20 pitfalls fixed during reproduction — each of which silently corrupts a number without raising — are documented in [`04_文档报告/执行记录-2026-10-20.md`](04_文档报告/执行记录-2026-10-20.md).

---

## Limitations

1. **The EOL metric is fragile**: MIT is truncated at EOL (no samples beyond crossing 80%), so zero-shot models that "just miss" score a constant 0%; NASA has 3 cells after gating and CALCE 8 — **neither supports significance testing**. Three replacement quantities (RUL relative error / signed overshoot / margin to crossing) turn the constant 0 into orderable numbers, but the sample size itself remains insufficient; EOL stays an auxiliary metric.
2. **Lag-Llama is capped at 32 context and is also the smallest model** (2.4 M measured weights, 1/19 of Chronos-T5-small): its native context is not comparable to the other models' 512, and the capacity gap is large. It is flagged separately, but it is still the worst on MIT (7.05), which means "crossing 80% 197 cycles early" cannot be attributed to context length or parameter count alone.
3. **Only two chemistries (LFP and LCO)**: no NCM/NCA, no fast-charging profiles, no temperature dimension.
4. **Few-shot fine-tuning covers Chronos-Bolt only**: the other four models' fine-tuning interfaces differ substantially and are not covered.
5. **Everything runs on CPU**: the full matrix takes about 10 hours, hence only 3 extra seeds (not 10).
6. **`knee_ratio` is a post-hoc quantity**: it needs ground truth, so it can only *explain* observed results, not serve as a deployment prior. Online proxies partially fill this gap — the knee is estimable from `drop_ctx` (ρ = −0.96), but model selection and per-cell interval width are not.
7. **The calibration head's `s` does not transfer across datasets**: MIT's `s = 1.65` is negative transfer on CALCE (+1.2%). `s` learns "how much to amplify this data distribution", not a model property, and must be refitted on the target distribution. The interval width multiplier `w` is likewise non-transferable, and context regression only reduces coverage error from 42pp to 19pp; conformal in-context self-testing is worse still (38pp deployable).
8. **The probabilistic path is far from calibrated**: even with the calibration head, PICP is only 25% (MIT) / 41% (NASA) / 17% (CALCE), nowhere near the nominal 80%. This benchmark can only point at the "overconfident" direction; it does not solve it.
9. **"Long context" is not measurable on this data**: none of the 183 cell × start combinations exceeds 2048 context (longest 1340), so the 512/1024/2048 comparison degenerates to "512 vs native". Studying context length requires a dataset with 4000+ cycles per cell (see [`04_文档报告/上下文范围.md`](04_文档报告/上下文范围.md)).
10. **Per-cell model selection has no headroom**: perfect selection among 6 candidates improves on the best single model by only 1.7% (MIT) / 0% (NASA) / 20% (CALCE), and leave-one-dataset-out learns no transferable rule. This is the ceiling for ensembling and routing approaches.
11. **Online estimation of interval width `w` is infeasible under this protocol**: feature regression reaches 19pp (LOOCV), conformal in-context self-testing is worse at 38pp deployable, and a post-hoc correction factor still loses to a constant baseline. The root cause is that the 50/50 split makes "a self-test matched to the horizon" mathematically non-existent, while `w` is extremely horizon-sensitive (halving the horizon leaves only 11.7% of `w`). Fixing it requires changing the protocol (a later start), which would change the definition of every published number (see [`04_文档报告/上下文自测.md`](04_文档报告/上下文自测.md)).

---

## Citation and license

Code is MIT, see [`LICENSE`](LICENSE). **Data and model weights are not distributed with this repository** — see [`NOTICE.md`](NOTICE.md) for distribution scope and access.

If this benchmark is useful to you:

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

[`CITATION.cff`](CITATION.cff) is also available (GitHub will show "Cite this repository").

---

*Last updated: 2026-10-22 · Short paper draft: [`04_文档报告/短文初稿-v1.md`](04_文档报告/短文初稿-v1.md) · Full execution log: [`04_文档报告/执行记录-2026-10-20.md`](04_文档报告/执行记录-2026-10-20.md)*
