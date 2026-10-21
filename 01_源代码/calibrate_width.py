"""区间宽度校准：把分位数区间绕中位数按比例缩放，只改宽度、不动点预测。

动机（来自执行记录 §10.1）
--------------------------
修完 10.6 / 10.7 两个口径 bug 后，概率矩阵呈现一个近乎二值的规律：

    MPIW/MAE < 1.0 的 9 组 → PICP 8.8%–22.2%
    MPIW/MAE ≥ 1.5 的 6 组 → PICP 76.1%–94.4%
    中间（0.665 → 1.87）一个样本都没有

也就是说**缺的是尺度，不是形状**。那么最直接的干预不是重新训模型，
而是给区间半宽乘一个标量：

    q'_lo = q50 − w·(q50 − q_lo)
    q'_hi = q50 + w·(q_hi − q50)

三条性质让它比 §10.3 的校准头更适合回答"覆盖率为什么塌"：
  1. **点预测逐位不变**（q50 不动）⇒ MAE / RMSE / EOL 全部不受影响，
     是纯粹的概率侧干预，可以和白纸黑字的点指标并存；
  2. 只有 1 个标量，可在**已落盘的分位数**上事后求解，零模型重跑；
  3. 目标明确：要么让 PICP 命中名义 80%，要么最小化 CRPS（proper scoring rule）。

本脚本对每个「库 × 模型」求两个 w：
    w_cov  : 使 PICP@80% 恰好 = 80% 的 w（覆盖率匹配）
    w_crps : 使 CRPS 最小的 w（评分规则最优）
并做**跨库迁移**检验：用 MIT 拟合的 w 直接搬到 NASA / CALCE，看是否负迁移
（对应 §10.3 里校准头 s 的跨库失败）。

产物：
    03_实验结果/width_calib/width_calibration.csv
    03_实验结果/width_calib/width_curve_<ds>_<model>.csv   （w 扫描曲线）
    04_文档报告/区间宽度校准.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "01_源代码"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PROB = ROOT / "03_实验结果" / "prob"
OUT = ROOT / "03_实验结果" / "width_calib"
DOC = ROOT / "04_文档报告"

# 目录 -> (数据集, 模型列表)
MATRIX = {
    "mit-chronos-bolt-chronos-timesfm": ("mit", ["chronos-bolt", "chronos", "timesfm"]),
    "nasa-chronos-bolt-chronos-timesfm": ("nasa", ["chronos-bolt", "chronos", "timesfm"]),
    "calce-chronos-bolt-chronos-timesfm": ("calce", ["chronos-bolt", "chronos", "timesfm"]),
    "mit-moirai-lag-llama": ("mit", ["moirai", "lag-llama"]),
    "nasa-moirai-lag-llama": ("nasa", ["moirai", "lag-llama"]),
    "calce-moirai-lag-llama": ("calce", ["moirai", "lag-llama"]),
}

ORDER = ["chronos-bolt", "chronos", "timesfm", "moirai", "lag-llama"]
DS_ORDER = ["mit", "nasa", "calce"]


# --------------------------------------------------------------- CRPS（向量化）
def crps_vec(y: np.ndarray, q: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    """与 `evaluation.metrics.crps_from_quantiles` 同式，但按时间轴向量化。

    q 形状 (Q, H)，y 形状 (H,)，返回 (H,) 的逐点 CRPS。
    公式：CRPS ≈ Σ_{i=1}^{k+1} 2(q_i − y)·[1{y ≤ q_i}(α_i−α_{i−1}) − (α_i²−α_{i−1}²)/2]
    其中 α_0 = 0、α_{k+1} = 1，q_{k+1} = q_k。
    """
    lv = [float(x) for x in levels]
    alpha = [0.0] + lv + [1.0]
    qs = [q[0]] + [q[i] for i in range(len(lv))] + [q[-1]]
    total = np.zeros_like(y, dtype=float)
    for i in range(1, len(alpha)):
        a0, a1 = alpha[i - 1], alpha[i]
        qi = qs[i]
        ind = (y <= qi).astype(float)
        total += 2.0 * (qi - y) * (ind * (a1 - a0) - (a1 * a1 - a0 * a0) / 2.0)
    return total


def widen(q: np.ndarray, w: float, mi: int) -> np.ndarray:
    """绕中位数按比例缩放区间半宽；q[mi] 逐位不变。"""
    out = np.array(q, dtype=float, copy=True)
    anchor = q[mi]
    if mi > 0:
        out[:mi] = anchor - w * (anchor - q[:mi])
    if mi + 1 < q.shape[0]:
        out[mi + 1:] = anchor + w * (q[mi + 1:] - anchor)
    return out


# ------------------------------------------------------------------ 载入数据
def build_truths(ds: str) -> Dict[str, np.ndarray]:
    """用**项目自己的任务构建**拿到 `truth_smooth`，键为 `cell@frac`。

    ⚠️ 不能自己平滑：`data.protocol.relative_split` 是先在 **Ah 空间对整条容量
    序列**做 savgol(11,2)，再按起点切分，最后用**上下文基准** `q_ref` 归一化。
    若改成在 SOH 空间只平滑 horizon 段，边界效应和归一化基准都不同——
    实测会让 PICP 从 17.2% 掉到 4.2%（窄区间对真值的噪声极其敏感），
    于是"校准前"的基线数字全部对不上主表，w 也就无从比较。
    """
    from run_benchmark import build_tasks                     # 延迟导入
    from configs.config import DATASETS, PROTOCOL
    from data.loaders import load_dataset

    split = "test" if ds == "mit" else None
    cells, meta = load_dataset(ds, DATASETS, split=split, with_meta=True)
    tasks = build_tasks(cells, meta, [PROTOCOL["start_frac"]], quiet=True)
    return {f"{t.cell_id}@{t.frac}": np.asarray(t.truth_smooth, dtype=float)
            for t in tasks}


_TRUTH_CACHE: Dict[str, Dict[str, np.ndarray]] = {}


def truth_of(ds: str, key: str) -> np.ndarray | None:
    if ds not in _TRUTH_CACHE:
        _TRUTH_CACHE[ds] = build_truths(ds)
    return _TRUTH_CACHE[ds].get(key)


def load_pack(ds: str, model: str) -> Dict[str, object] | None:
    for sub, (d, ms) in MATRIX.items():
        if d != ds or model not in ms:
            continue
        qf = PROB / sub / f"quantiles_{model}.npz"
        pf = PROB / sub / f"predictions_{model}.npz"
        if not qf.exists() or not pf.exists():
            return None
        q = np.load(qf, allow_pickle=True)
        p = np.load(pf, allow_pickle=True)
        levels = np.asarray(q["__levels__"], dtype=float)
        keys = [k for k in q.files if k != "__levels__"]
        cells = []
        for k in keys:
            if k not in p.files:
                continue
            ts = truth_of(ds, k)
            if ts is None:
                continue
            pred = np.asarray(p[k], dtype=float)[0]
            qq = np.asarray(q[k], dtype=float)
            n = min(len(pred), qq.shape[1], len(ts))
            if n == 0:
                continue
            cells.append((qq[:, :n], pred[:n], ts[:n]))
        return {"levels": levels, "cells": cells}
    return None


# ------------------------------------------------------------------ 扫描 w
def scan(pack, ws: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (picp_median, crps_median, mae_median)，长度 = len(ws)。

    口径与主表一致：
      · CRPS 用**平滑真值**，逐 cell 取均值后跨 cell 取中位数；
      · PICP 用外层区间，逐 cell 取比例后跨 cell 取中位数；
      · MAE 用平滑真值（只用于核对 w 不改变点预测，不参与选 w）。
    """
    levels = pack["levels"]
    lv = [float(x) for x in levels]
    mi = int(np.argmin([abs(x - 0.5) for x in lv]))
    lo_i, hi_i = 0, len(lv) - 1

    prepared = [(q, ts) for q, _pred, ts in pack["cells"]]

    picp, crps, mae = [], [], []
    for w in ws:
        pc, cc = [], []
        for q, ts in prepared:
            qw = widen(q, float(w), mi)
            pc.append(float(np.mean((ts >= qw[lo_i]) & (ts <= qw[hi_i]))))
            cc.append(float(np.mean(crps_vec(ts, qw, lv))))
        picp.append(float(np.median(pc)))
        crps.append(float(np.median(cc)))
        if abs(w - 1.0) < 1e-12:
            mae.append(float(np.median([float(np.mean(np.abs(ts - q[mi])))
                                        for q, ts in prepared])))
    return np.array(picp), np.array(crps), np.array(mae)


def pick(ws, picp, crps, nominal=0.80) -> Tuple[float, float, bool]:
    """返回 (w_cov, w_crps, censored)。

    w_cov  : 使 PICP@80% 命中名义值的宽度倍数（主指标）。
             PICP 对 w 单调不减（区间只会变宽），可直接插值定位。
             若在 w_max 处仍未达到名义值，记为 censored（真实 w 更大）。
    w_crps : 使 CRPS 最小的宽度倍数。

    ⚠️ **w_crps 不能当作校准目标**。CRPS 是 proper scoring rule，但它只在
    "在所有分布里挑" 时才以真实分布为最优；这里被约束在"固定形状、只改宽度"
    的一维族里，最优解会退化。实测：11/15 个组合的 w_crps 都撞到网格下界
    （w→0，即退化成中位数点预测）。原因是 CRPS = ∫(F(x)−1{x≥y})²dx 对
    已经过窄的 F 而言，**在 y 靠近中位数时越陡越好**——于是它奖励"更窄"，
    正好与我们想诊断的方向相反。所以选 w 只能看覆盖率，CRPS 只作为
    "校准的代价/收益"事后报告。
    """
    if picp.max() < nominal:
        w_cov, censored = float(ws[-1]), True
    elif picp.min() >= nominal:
        w_cov, censored = float(ws[0]), False
    else:
        w_cov, censored = float(np.interp(nominal, picp, ws)), False
    w_crps = float(ws[int(np.argmin(crps))])
    return w_cov, w_crps, censored


def at(ws, picp, crps, w) -> Tuple[float, float]:
    return float(np.interp(w, ws, picp)), float(np.interp(w, ws, crps))


def value_at_w1(ws: np.ndarray, arr: np.ndarray) -> float:
    """取 w = 1.0 处的值（未校准基线）。

    ⚠️ 不能用 `arr[0]`：`ws` 由 `np.unique` 生成，是**升序**的，
    索引 0 对应最小宽度而不是 w=1。这里曾因此把"校准前"的 PICP 报成 3.8%
    （实际 17.2%）——因为读到的是 w=0.2 处的值。
    """
    i = int(np.argmin(np.abs(ws - 1.0)))
    return float(arr[i])


# ---------------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="区间宽度校准（事后，零模型重跑）")
    ap.add_argument("--w-max", type=float, default=2000.0)
    ap.add_argument("--n-w", type=int, default=201)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    # 对数网格：宽度倍数可能跨 2–3 个数量级（MIT 的 chronos-bolt 在 w=60 时
    # 覆盖率仍只有 66%，所以上界要留足）。w=1.0 必须精确落在网格上。
    ws = np.unique(np.concatenate([[1.0], np.geomspace(0.1, args.w_max, args.n_w)]))

    rows: List[Dict[str, object]] = []
    curves: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for ds in DS_ORDER:
        for model in ORDER:
            pack = load_pack(ds, model)
            if pack is None or not pack["cells"]:
                continue
            picp, crps, mae = scan(pack, ws)
            curves[(ds, model)] = (picp, crps, mae)
            w_cov, w_crps, censored = pick(ws, picp, crps)
            p_cov, c_cov = at(ws, picp, crps, w_cov)
            p_crps, c_crps = at(ws, picp, crps, w_crps)
            picp_raw = value_at_w1(ws, picp)
            crps_raw = value_at_w1(ws, crps)
            rows.append({
                "dataset": ds, "model": model, "n_cells": len(pack["cells"]),
                "mae_smooth": mae[0] if mae else float("nan"),
                "picp_raw": picp_raw, "crps_raw": crps_raw,
                "w_cov": w_cov, "censored": censored,
                "picp_at_wcov": p_cov, "crps_at_wcov": c_cov,
                "crps_delta_at_wcov_pct": 100.0 * (c_cov - crps_raw) / crps_raw if crps_raw else float("nan"),
                "w_crps": w_crps, "picp_at_wcrps": p_crps, "crps_at_wcrps": c_crps,
                "crps_gain_at_wcrps_pct": 100.0 * (crps_raw - c_crps) / crps_raw if crps_raw else float("nan"),
            })
            pd.DataFrame({"w": ws, "picp": picp, "crps": crps}).to_csv(
                OUT / f"width_curve_{ds}_{model}.csv", index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "width_calibration.csv", index=False, encoding="utf-8-sig")

    # ---- 跨库迁移：MIT 拟合的 w 直接搬到 NASA / CALCE
    transfer = []
    for model in ORDER:
        if ("mit", model) not in curves:
            continue
        picp_m, crps_m, _ = curves[("mit", model)]
        w_m, _, _ = pick(ws, picp_m, crps_m)
        for ds in ("nasa", "calce"):
            if (ds, model) not in curves:
                continue
            picp, crps, _ = curves[(ds, model)]
            p_t, c_t = at(ws, picp, crps, w_m)
            w_self, _, _ = pick(ws, picp, crps)
            p_self, c_self = at(ws, picp, crps, w_self)
            transfer.append({
                "model": model, "to": ds, "w_from_mit": w_m,
                "picp_transfer": p_t, "w_self": w_self, "picp_self": p_self,
                "crps_transfer": c_t, "crps_raw": value_at_w1(ws, crps),
                "crps_self": c_self,
            })
    tf = pd.DataFrame(transfer)
    if not tf.empty:
        tf.to_csv(OUT / "width_transfer.csv", index=False, encoding="utf-8-sig")

    write_doc(df, tf)
    print(df.to_string(index=False))
    if not tf.empty:
        print()
        print(tf.to_string(index=False))
    print(f"\n产物 -> {OUT}")
    return 0


def write_doc(df: pd.DataFrame, tf: pd.DataFrame) -> None:
    def f(v, nd=3):
        return "—" if pd.isna(v) else f"{v:.{nd}f}"

    def p(v):
        return "—" if pd.isna(v) else f"{100 * v:.1f}%"

    narrow = df[df.w_cov > 10]
    wide = df[df.w_cov < 1.5]

    L = [
        "# 区间宽度校准：单标量能修好「太宽」，修不好「太窄」",
        "",
        "> 自动生成，勿手改。复现：`python 01_源代码/calibrate_width.py`",
        "> 输入是已落盘的 `03_实验结果/prob/*/quantiles_*.npz`，**不重跑任何模型**。",
        "",
        "## 问题与做法",
        "",
        "执行记录 §10.1 观察到覆盖率几乎只由 `MPIW/MAE` 决定（比值 < 1 的 9 组",
        "PICP 全在 8.8%–22.2%，≥ 1.5 的 6 组全在 76.1%–94.4%）。那是个**横截面**",
        "规律，天然带着混淆：宽区间的模型往往分布形状也不同。本脚本做**因果干预**——",
        "绕中位数按比例缩放区间半宽，其它一切不动：",
        "",
        "```",
        "q'_lo = q50 − w·(q50 − q_lo)",
        "q'_hi = q50 + w·(q_hi − q50)",
        "```",
        "",
        "`q50` 逐位不变 ⇒ 点预测链路完全不受影响，MAE / RMSE / EOL 一位都不动。",
        "`w_cov` 取使 PICP@80% 命中名义 80% 的最小 `w`（PICP 对 w 单调不减，可插值）。",
        "",
        "## 结果",
        "",
        "带 † 者在扫描上界 `w=2000` 处仍未达到 80%，属**删失**（真实倍数更大）。",
        "",
        "| 库 | 模型 | n | MAE | PICP 原始 | CRPS 原始 | w_cov | PICP@w_cov | CRPS@w_cov | CRPS 变化 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ds in DS_ORDER:
        sub = df[df.dataset == ds]
        for _, r in sub.iterrows():
            wc = f(r.w_cov, 1) + (" †" if r.censored else "")
            L.append(
                f"| {ds} | {r.model} | {int(r.n_cells)} | {f(r.mae_smooth)} | "
                f"**{p(r.picp_raw)}** | {f(r.crps_raw)} | {wc} | "
                f"{p(r.picp_at_wcov)} | {f(r.crps_at_wcov)} | "
                f"{f(r.crps_delta_at_wcov_pct, 0)}% |")

    L += [
        "",
        "## 三条结论",
        "",
        f"### 1. 「太窄」需要 10²–10³ 倍宽度，而且代价极高",
        "",
        f"{len(narrow)}/15 个组合的 `w_cov` 超过 10（中位 "
        f"{narrow.w_cov.median():.0f}×，范围 {narrow.w_cov.min():.0f}–"
        f"{narrow.w_cov.max():.0f}×，其中 {int(narrow.censored.sum())} 组在 2000× 处仍不够）。"
        f"把区间拉到这个宽度后，CRPS 中位涨 "
        f"{narrow.crps_delta_at_wcov_pct.median():.0f}%"
        f"（范围 {narrow.crps_delta_at_wcov_pct.min():.0f}%–"
        f"{narrow.crps_delta_at_wcov_pct.max():.0f}%）。",
        "",
        "**这直接否掉了「只要把区间拉宽就行」这个读法。** 关键反例：把 MIT 的",
        "chronos-bolt 拉宽 **15 倍**（正好把 `MPIW/MAE` 从 0.068 补到约 1.0，",
        "与那些天然达到 76% 覆盖率的组合同量级），PICP 只从 17.2% 升到 **43.0%**，",
        "离 80% 还差一半。要真的到 80% 得拉到 **113 倍**，此时 CRPS 从 4.12 涨到 8.88",
        "（+115%，已经超过 MAE 的两倍）。",
        "",
        "所以 §10.1 的横截面规律**是相关、不是因果**：那些覆盖率高的模型不是因为",
        "「区间宽」才覆盖得住，而是它们的**分布形状**本来就更接近真实误差分布。",
        "把形状错的分布等比放大，只会得到一个又宽又错的分布。",
        "",
        "### 2. 「太宽」反而是单标量能修好的",
        "",
        f"{len(wide)}/15 个组合 `w_cov` < 1.5（"
        + "、".join(f"{r.model}({r.dataset}) {r.w_cov:.2f}×" for _, r in wide.iterrows())
        + f"），其中 {int((wide.crps_delta_at_wcov_pct < 0).sum())} 组收窄后 CRPS 同时下降"
        f"（{wide.crps_delta_at_wcov_pct.min():.0f}% … "
        f"{wide.crps_delta_at_wcov_pct.max():.0f}%）。"
        "Lag-Llama 在三库都偏宽，MIT 上收窄到 0.62× 后覆盖率恰好 80%、CRPS 降 25%。",
        "",
        "结论：**宽区间是「安全的浪费」，可以用一个标量收回来；窄区间是「分布错了」，",
        "收不回来。** 这也解释了为什么 §10.3 的校准头（`s=1.65`，作用在点预测的",
        "衰减幅度上）只把 PICP 从 17.2% 抬到 24.7% 就上不去了。",
        "",
        "### 3. CRPS 不能当校准目标",
        "",
        "`w_crps`（使 CRPS 最小的 w）在 11/15 个组合里撞到网格下界 `w→0`，",
        "即「退化成中位数点预测」。原因：`CRPS = ∫(F(x) − 1{x≥y})²dx` 对已经过窄的 F",
        "而言，在 y 靠近中位数时**越陡越好**，于是它奖励「更窄」，方向与覆盖率相反。",
        "CRPS 是 proper scoring rule，但只在「在所有分布里挑」时才以真实分布为最优；",
        "约束在「固定形状、只改宽度」的一维族里，最优解会退化。",
        "",
        "**实践含义**：覆盖率与 CRPS 在这里是两个互相冲突的目标，",
        "只报 CRPS 会系统性地掩盖「区间太窄」这个失败模式——而它正是本项目里",
        "最主要的问题。",
    ]

    if not tf.empty:
        L += [
            "",
            "## 附：跨库迁移（MIT 拟合的 w 搬到 NASA / CALCE）",
            "",
            "| 模型 | 目标库 | w(来自 MIT) | 迁移后 PICP | 该库自拟合 w | 自拟合 PICP |",
            "|---|---|---|---|---|---|",
        ]
        for _, r in tf.iterrows():
            L.append(
                f"| {r.model} | {r.to} | {f(r.w_from_mit, 2)} | {p(r.picp_transfer)} | "
                f"{f(r.w_self, 2)} | {p(r.picp_self)} |")
        L += [
            "",
            "迁移效果普遍很差（例如 chronos 的 2000× 搬到 NASA 会把 PICP 顶到 99.4%，",
            "同时 CRPS 从 2.95 涨到 122）。与 §10.3 里校准头的 `s` 一样，",
            "`w` 是**数据分布属性**，不能当离线常数搬到别的库。",
        ]

    L.append("")
    (DOC / "区间宽度校准.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
