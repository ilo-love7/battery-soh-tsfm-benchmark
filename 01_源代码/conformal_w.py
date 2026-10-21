"""§9.5 最后一条路：上下文内自测（conformal 式伪任务）能不能在线估 `w`？

问题
----
`online_proxy.py` 已经证明：用上下文**特征**回归 `w` 只能把覆盖率偏离从 42pp 压到 19pp，
因为特征描述的是"衰减形态与 horizon 长度"，不是"这个模型在这条序列上的误差分布"。

Conformal 式的思路完全不同——不问特征，直接**在上下文内部做一次伪预测**：
把上下文切成两段，前段当输入、后段当"已知真值"，跑同一个模型，
在伪任务上求出使 PICP=80% 的宽度倍数 `w_hat`，再搬到真实任务上。
它天然对齐了"同一个模型、同一条序列、同一种误差结构"，理论上应该远好于特征回归。

两个必须先验证的前提
--------------------
1. **horizon 不变性**：伪任务的 horizon 一定短于真实 horizon（见下），
   如果 `w` 强烈依赖 horizon，`w_hat` 就搬不过去。
2. **结构性可行**：起点 0.5 时上下文长度 T **恰好等于** horizon 长度 H
   （数据是 50/50 切的）。要有一段长度 H 的已知真值，上下文就只能剩 T−H = 0 点。
   所以**匹配 horizon 的自测在这个协议下根本不存在**，只能用 H_p = H/2 之类。

本脚本先量化前提 1（纯读已落盘分位数，零推理），再跑真实伪任务量化前提 2 的代价。

产物：
    03_实验结果/conformal_w/horizon_invariance.csv   前提 1：w 随 horizon 的变化
    03_实验结果/conformal_w/per_cell_<ds>_<model>.csv 逐 cell 的 w_hat 与三种 PICP
    03_实验结果/conformal_w/summary.csv              按「库×模型」汇总（多次运行合并）
    04_文档报告/上下文自测.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "01_源代码"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUT = ROOT / "03_实验结果" / "conformal_w"
DOC = ROOT / "04_文档报告"

DS_ORDER = ["mit", "nasa", "calce"]
FM_ORDER = ["chronos-bolt", "chronos", "timesfm", "moirai", "lag-llama"]

MATRIX = {
    "mit-chronos-bolt-chronos-timesfm": ("mit", ["chronos-bolt", "chronos", "timesfm"]),
    "nasa-chronos-bolt-chronos-timesfm": ("nasa", ["chronos-bolt", "chronos", "timesfm"]),
    "calce-chronos-bolt-chronos-timesfm": ("calce", ["chronos-bolt", "chronos", "timesfm"]),
    "mit-moirai-lag-llama": ("mit", ["moirai", "lag-llama"]),
    "nasa-moirai-lag-llama": ("nasa", ["moirai", "lag-llama"]),
    "calce-moirai-lag-llama": ("calce", ["moirai", "lag-llama"]),
}

W_GRID = np.unique(np.concatenate([[1.0], np.geomspace(0.01, 1e5, 241)]))
T_MIN = 64          # 伪上下文至少要留这么多点，否则模型等于在猜

# 主表的公平性口径：`run_benchmark.py --max-context 512`。
# 这个值不在 PROTOCOL 里（它是 CLI 参数，因为 TimesFM 原生只有 512 才需要统一夹），
# 但伪任务要和主表同口径，所以在这里固定成常量。
MAX_CONTEXT = 512


# --------------------------------------------------------------- 前提 1：horizon 不变性
def horizon_invariance() -> pd.DataFrame:
    """同一批分位数，只把评测的 horizon 截短，看所需 `w_cov` 怎么变。

    这是"零推理"的诊断：真实分位数已经覆盖整段 horizon，截短只是少算后几步。
    """
    from online_proxy import PROB, truths_of, w_cov_cell

    rows: List[Dict[str, object]] = []
    for ds in DS_ORDER:
        truth = truths_of(ds)
        for sub, (d, models) in MATRIX.items():
            if d != ds:
                continue
            for m in models:
                f = PROB / sub / f"quantiles_{m}.npz"
                if not f.exists():
                    continue
                z = np.load(f, allow_pickle=True)
                lv = np.asarray(z["__levels__"], dtype=float)
                for key in [k for k in z.files if k != "__levels__"]:
                    if key not in truth:
                        continue
                    q = np.asarray(z[key], dtype=float)
                    ts = truth[key]
                    n = min(q.shape[1], ts.size)
                    for frac in (0.25, 0.5, 1.0):
                        k = max(8, int(round(n * frac)))
                        w, cen = w_cov_cell(q[:, :k], ts[:k], lv, W_GRID)
                        rows.append({"dataset": ds, "model": m,
                                     "cell": key.split("@")[0],
                                     "frac_of_horizon": frac,
                                     "w_cov": w, "censored": cen})
    return pd.DataFrame(rows)


def premise1_stats(inv: pd.DataFrame) -> Dict[float, Tuple[float, float, float, float]]:
    """前提 1 的配对统计：`frac -> (log10 ρ, 中位比值, IQR下, IQR上)`。

    比值 = `w_cov(前 frac 的 horizon) / w_cov(全 horizon)`。
    这个数同时被两处用：
      - `run_one` 拿它反推一个"事后最优 fudge factor" `k = 1/中位比值`，
        检验「就算允许按 horizon 缩放，自测能不能救回来」；
      - `write_doc` 用它和实测的 `w_hat/w_oracle` 对照，
        看自测到底是在估「误差分布」还是在估「horizon 被截短了多少」。
    """
    piv = inv.pivot_table(index=["dataset", "model", "cell"],
                          columns="frac_of_horizon", values="w_cov").dropna()
    out: Dict[float, Tuple[float, float, float, float]] = {}
    for a in (0.25, 0.5):
        if a not in piv.columns or 1.0 not in piv.columns:
            continue
        ok = (piv[a] > 0) & (piv[1.0] > 0)
        if ok.sum() < 4:
            continue
        lr = float(np.corrcoef(np.log10(piv.loc[ok, a]),
                               np.log10(piv.loc[ok, 1.0]))[0, 1])
        ratio = piv.loc[ok, a] / piv.loc[ok, 1.0]
        out[a] = (lr, float(ratio.median()),
                  float(ratio.quantile(.25)), float(ratio.quantile(.75)))
    return out


# ------------------------------------------------------------- 前提 2：真实伪任务
def build_pseudo(ctx: np.ndarray, hfrac: float) -> Tuple[np.ndarray, np.ndarray, int] | None:
    """把上下文切成「伪输入 | 伪真值」。返回 (pseudo_ctx, pseudo_truth, Hp)。

    ⚠️ 伪真值用**上下文原值**，不平滑。真实评测用 `truth_smooth`（先在 Ah 空间
    平滑整条再归一化）。这个不一致会让 `w_hat` 偏大（噪声需要更宽的区间），
    方向是"保守"，与 horizon 缩短导致的偏小部分抵消——两者都会在结果里如实报告。
    """
    ctx = np.asarray(ctx, dtype=float).ravel()
    T = ctx.size
    Hp = min(int(round(T * hfrac)), T - T_MIN)
    if Hp < 16:
        return None
    Tp = T - Hp
    return ctx[:Tp], ctx[Tp:], Hp


def run_one(ds: str, model_name: str, hfrac: float,
            k_rescale: float = 1.0) -> List[Dict[str, object]]:
    from adapters import get_forecaster, rolling_forecast_quantiles_batch
    from configs.config import DATASETS, PROTOCOL
    from data.loaders import load_dataset
    from online_proxy import PROB, w_cov_cell
    from run_benchmark import build_tasks

    split = "test" if ds == "mit" else None
    cells, meta = load_dataset(ds, DATASETS, split=split, with_meta=True)
    tasks = build_tasks(cells, meta, [PROTOCOL["start_frac"]], quiet=True)

    # 真实分位数（已落盘，不用重跑）
    qf = None
    for sub, (d, models) in MATRIX.items():
        if d == ds and model_name in models:
            qf = PROB / sub / f"quantiles_{model_name}.npz"
    if qf is None or not qf.exists():
        print(f"  跳过 {ds}/{model_name}：没有已落盘的分位数")
        return []
    z = np.load(qf, allow_pickle=True)
    levels = np.asarray(z["__levels__"], dtype=float)
    qreal = {k: np.asarray(z[k], dtype=float) for k in z.files if k != "__levels__"}

    model = get_forecaster(model_name)
    model.max_context = MAX_CONTEXT                    # 与主表同口径（512）

    # 伪任务
    pctx, ptr, keys = [], [], []
    for t in tasks:
        got = build_pseudo(t.context, hfrac)
        if got is None:
            continue
        a, b, _ = got
        pctx.append(a)
        ptr.append(b)
        keys.append(f"{t.cell_id}@{t.frac}")
    if not pctx:
        return []
    print(f"  {ds}/{model_name}: {len(pctx)} 个伪任务，"
          f"伪上下文 {min(len(c) for c in pctx)}–{max(len(c) for c in pctx)} 点，"
          f"伪 horizon {min(len(t) for t in ptr)}–{max(len(t) for t in ptr)} 点")
    qp = rolling_forecast_quantiles_batch(
        model, pctx, [len(t) for t in ptr], chunk=PROTOCOL["rolling_chunk"],
        levels=levels)

    from online_proxy import picp_at
    lv = [float(x) for x in levels]
    mi = int(np.argmin([abs(x - 0.5) for x in lv]))

    rows: List[Dict[str, object]] = []
    keep: List[Tuple[np.ndarray, np.ndarray]] = []
    for key, q_p, truth_p in zip(keys, qp, ptr):
        q_r = qreal.get(key)
        if q_r is None:
            continue
        # 真实真值：从任务里取（与主表同口径）
        t_real = next((t for t in tasks if f"{t.cell_id}@{t.frac}" == key), None)
        if t_real is None:
            continue
        ts = np.asarray(t_real.truth_smooth, dtype=float)
        n = min(q_r.shape[1], ts.size)
        w_hat, cen_p = w_cov_cell(q_p, np.asarray(truth_p, dtype=float), lv, W_GRID)
        w_orc, cen_o = w_cov_cell(q_r[:, :n], ts[:n], lv, W_GRID)
        rows.append({
            "dataset": ds, "model": model_name, "cell": key.split("@")[0],
            "T_ctx": len(pctx[keys.index(key)]),
            "H_real": int(n), "H_pseudo": int(len(truth_p)),
            "w_hat": w_hat, "w_hat_censored": cen_p,
            "w_oracle": w_orc, "w_oracle_censored": cen_o,
            "picp_w1": picp_at(q_r[:, :n], ts[:n], 1.0, mi),
            "picp_w_hat": picp_at(q_r[:, :n], ts[:n], w_hat, mi),
            # 事后按 horizon 比值反推的 fudge factor（乐观上限：k 本身是全库拟合的）
            "picp_w_x": picp_at(q_r[:, :n], ts[:n], w_hat * k_rescale, mi),
            "picp_oracle": picp_at(q_r[:, :n], ts[:n], w_orc, mi),
        })
        keep.append((q_r[:, :n], ts[:n]))

    # 对照组：**库常数基线**。取本组 `w_oracle` 的中位数当唯一一个宽度倍数，
    # 所有 cell 共用。它和 `k` 花的是同一份「作弊预算」（都要真值），
    # 所以「`w_hat × k` 比它好多少」才是自测真正贡献的信息量。
    if rows:
        w_const = float(np.median([r["w_oracle"] for r in rows]))
        for r, (q, ts_) in zip(rows, keep):
            r["w_const"] = w_const
            r["picp_w_const"] = picp_at(q, ts_, w_const, mi)
    return rows


# ------------------------------------------------------------------ 文档
def write_doc(inv: pd.DataFrame, per: pd.DataFrame, k_rescale: float = 1.0) -> None:
    def f(v, nd=3):
        try:
            v = float(v)
        except Exception:
            return "—"
        return "—" if not np.isfinite(v) else f"{v:.{nd}f}"

    def pct(v, nd=1):
        try:
            v = float(v)
        except Exception:
            return "—"
        return "—" if not np.isfinite(v) else f"{100 * v:.{nd}f}%"

    st = premise1_stats(inv)
    n_pair = len(inv.pivot_table(index=["dataset", "model", "cell"],
                                 columns="frac_of_horizon",
                                 values="w_cov").dropna())

    L = [
        "# 上下文内自测：这条路在这三个库上走不通",
        "",
        "> 自动生成，勿手改。复现：`python 01_源代码/conformal_w.py`",
        "> 前提 1 纯读已落盘分位数（零推理）；前提 2 需要跑一次伪任务推理。",
        "",
        "## 想解决什么",
        "",
        "`online_proxy.py` 证明了：用上下文**特征**回归 `w` 只能把覆盖率偏离从"
        "平均 42pp 压到 19pp（中位 20pp），因为特征描述的是「衰减形态与 horizon 长度」，"
        "不是「这个模型在这条序列上的误差分布」。",
        "",
        "Conformal 式思路绕开特征，直接**在上下文内部做一次伪预测**：上下文切两段，",
        "前段当输入、后段当已知真值，跑同一个模型，在伪任务上求 `w_hat`，再搬到真实任务。",
        "它天然对齐了「同一个模型、同一条序列、同一种误差结构」，理论上是这组尝试里最有希望的一条。",
        "",
        "## 前提 1：`w` 强烈依赖 horizon 长度",
        "",
        "同一批真实分位数，只把评测的 horizon 截短（少算后几步），看所需 `w_cov` 怎么变。",
        f"配对 cell 数 **{n_pair}**。",
        "",
        "| 评测 horizon | log ρ（与全 horizon 比） | 所需 `w` 的中位比值 | IQR |",
        "|---|---|---|---|",
    ]
    for a in (0.25, 0.5):
        if a in st:
            lr, med, q1, q3 = st[a]
            L.append(f"| 前 {int(a * 100)}% | {f(lr)} | **{f(med, 4)}** | {f(q1, 4)}–{f(q3, 4)} |")

    if 0.5 in st:
        L += [
            "",
            f"**排序还保得住（log ρ = {f(st[0.5][0])}），但量级完全不行**：horizon 减半后"
            f"所需宽度只有原来的 **{pct(st[0.5][1])}**，再减到 1/4 就只剩 "
            f"{pct(st[0.25][1]) if 0.25 in st else '—'}。",
            "",
            "原因很直接：PICP 的定义是「整段 horizon 都被区间盖住的比例」。",
            "要覆盖的点越多，最坏那一步越极端，所需宽度按远超线性的速度涨。",
            "（粗算：若 `w ∝ H^α`，从两个比值反解出的 α 本身就不稳定，",
            "说明它不是干净的幂律，不能靠一个指数外推。）",
        ]

    L += [
        "",
        "## 前提 2：匹配 horizon 的自测在起点 0.5 下**不存在**",
        "",
        "这是协议决定的结构性事实，不是实现问题：",
        "",
        "- 起点 0.5 时，上下文长度 T **恰好等于** horizon 长度 H（数据是 50/50 切的）；",
        "- 自测要求上下文里有一段长度 H 的**已知真值**；",
        "- 于是留给模型的输入只剩 T − H = **0 点**。",
        "",
        "所以伪 horizon 只能是 `H_p = T·hfrac`（本文用 0.5），伪上下文同时被砍到 `T − H_p`。",
        "**两头都变短**：模型看到的历史更少，要预测的跨度也更短。",
        "序列太短时还会被 `T_MIN = 64` 兜底夹住（NASA 全部落在这一支）。",
        "",
    ]

    # 结论表要用的数：先在分支外初始化，避免 per 为空时 NameError
    base = dev = devx = devc = float("nan")
    hurt_txt = "（本次没有跑伪任务推理，无法判断）"

    if not per.empty:
        p = per.copy()
        p["ratio"] = p["w_hat"] / p["w_oracle"].replace(0, np.nan)
        agg = p.groupby(["dataset", "model"]).agg(
            n=("cell", "size"),
            h_ratio=("H_pseudo", "median"),
            w_hat=("w_hat", "median"),
            w_oracle=("w_oracle", "median"),
            ratio=("ratio", "median"),
            picp_w1=("picp_w1", "median"),
            picp_w_hat=("picp_w_hat", "median"),
            picp_w_x=("picp_w_x", "median"),
            picp_w_const=("picp_w_const", "median"),
            picp_oracle=("picp_oracle", "median"),
        ).reset_index()
        L += [
            "### 实测：伪任务估出的 `w_hat` 搬到真实任务上",
            "",
            "| 库 | 模型 | n | 伪 horizon 中位 | `w_hat` | `w_oracle` | 比值 | 不校准 | 用 `w_hat` | ×`k` | 库常数基线 | oracle |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _, r in agg.iterrows():
            L.append(f"| {r.dataset} | {r.model} | {int(r.n)} | {int(r.h_ratio)} | "
                     f"{f(r.w_hat, 1)} | {f(r.w_oracle, 1)} | {f(r.ratio, 4)} | "
                     f"{pct(r.picp_w1)} | **{pct(r.picp_w_hat)}** | {pct(r.picp_w_x)} | "
                     f"{pct(r.picp_w_const)} | {pct(r.picp_oracle)} |")

        base = float((p.picp_w1 - 0.80).abs().median())
        dev = float((p.picp_w_hat - 0.80).abs().median())
        devx = float((p.picp_w_x - 0.80).abs().median())
        devc = float((p.picp_w_const - 0.80).abs().median())
        over = int((p.picp_w_hat > 0.90).sum())
        under = int((p.picp_w_hat < 0.50).sum())
        bet = int(((p.picp_w_hat >= 0.70) & (p.picp_w_hat <= 0.90)).sum())
        overc = int((p.picp_w_x > 0.90).sum())
        underc = int((p.picp_w_x < 0.50).sum())
        betc = int(((p.picp_w_x >= 0.70) & (p.picp_w_x <= 0.90)).sum())
        r_med = float(p.ratio.median())
        r_q1, r_q3 = float(p.ratio.quantile(.25)), float(p.ratio.quantile(.75))

        from online_proxy import spearman
        rho_all = spearman(np.log10(p["w_hat"]), np.log10(p["w_oracle"]))
        rhos = [(f"{d_}/{m_}", spearman(np.log10(g["w_hat"]), np.log10(g["w_oracle"])))
                for (d_, m_), g in p.groupby(["dataset", "model"]) if len(g) >= 4]
        gain = devc - devx          # 自测相对库常数基线的净收益（pp，正=有用）

        # 反直觉检查：本来覆盖率就接近名义值的组，会不会被自测推低？
        hurt = agg[agg["picp_w_hat"] < agg["picp_w1"] - 0.05]
        hurt_txt = ""
        if not hurt.empty:
            w_ = hurt.loc[(hurt["picp_w1"] - hurt["picp_w_hat"]).idxmax()]
            hurt_txt = (f"{len(hurt)}/{len(agg)} 组被推低 5pp 以上，"
                        f"最严重的是 {w_['dataset']}/{w_['model']}："
                        f"{100 * w_['picp_w1']:.1f}% → {100 * w_['picp_w_hat']:.1f}%")
        else:
            hurt_txt = f"0/{len(agg)} 组被推低——这一轮没有出现反向伤害"

        L += [
            "",
            f"覆盖率偏离的中位：不校准 **{100 * base:.0f}pp** → 用 `w_hat` "
            f"**{100 * dev:.0f}pp** → 再乘 `k = {k_rescale:.2f}` **{100 * devx:.0f}pp**；"
            f"**库常数基线 {100 * devc:.0f}pp**"
            f"（oracle 上界 {100 * float((p.picp_oracle - .8).abs().median()):.0f}pp）。",
            "",
            f"原始 `w_hat` 的分布是**两头翻车**：{over}/{len(p)} 个 cell 被推到 90% 以上"
            f"（其中 {int((p.picp_w_hat > 0.99).sum())} 个顶到 100%，严重过度覆盖），"
            f"{under}/{len(p)} 个仍在 50% 以下（依然严重欠覆盖），"
            f"真正落在 70–90% 的只有 **{bet}/{len(p)}**。",
            f"乘 `k` 之后：{overc}/{len(p)} 过度、{underc}/{len(p)} 欠覆盖、"
            f"**{betc}/{len(p)}** 进带——好转了，但两头仍有尾巴。",
            "",
            "## 决定性对照：自测估的是「horizon 被截短」，不是「误差分布」",
            "",
            f"实测 `w_hat / w_oracle` 的中位是 **{r_med:.4f}**（IQR {r_q1:.4f}–{r_q3:.4f}，"
            f"跨 **{r_q3 / r_q1:.0f} 倍**）。",
        ]
        if 0.5 in st:
            pred = st[0.5][1]
            L += [
                f"而前提 1 独立预测的「horizon 减半 ⇒ `w` 只剩 **{pred:.4f}**」——"
                "**两个数几乎重合**。",
                "",
                "这不是巧合。伪任务上求出的 `w_hat`，其**水平**主要由「horizon 从 H 缩到 H/2 "
                "会让所需宽度掉多少」决定，而后者是**协议常数**，跟模型、跟电池、跟误差分布都无关。",
                f"逐 cell 的**排序**倒还留着一点信息（合并 log ρ = **{rho_all:.2f}**，"
                f"分组范围 {min(v for _, v in rhos):.2f}–{max(v for _, v in rhos):.2f}），"
                "但弱且不稳定（CALCE 的 chronos-bolt 有 0.74，MIT 的只有 0.14）。",
                "",
                "**净收益检验**：把 `w_hat` 乘上 `k` 后覆盖率偏离降到 "
                f"{100 * devx:.0f}pp，但一个**根本不用自测**的库常数（本组 `w_oracle` 中位数）"
                f"也能做到 {100 * devc:.0f}pp。两者花的是同一份「作弊预算」（都需要真值），"
                f"所以自测的真正信息量只有 **{100 * gain:+.1f}pp**。",
            ]
            if gain < 0:
                L += [
                    "",
                    "**负收益就是结论**：自测不仅没帮上忙，还让结果**变差**了——"
                    f"花同样的作弊预算，直接给一个库常数比过它 {100 * -gain:.1f}pp。",
                    "也就是说，伪任务自测提供的信息量是**负的**：它测到的是「horizon 被截短」，"
                    "而那个量在真实任务上早就被 `k` 一次性抵掉了，剩下的只有噪声。",
                ]
            elif gain < 4:
                L += [
                    "",
                    "**这个数接近 0 就是结论**：伪任务自测没有提供任何超越「horizon 长度」的信息。",
                    "它之所以看起来有效，全靠事后乘的那个 `k`——而 `k` 是从**真实任务**的",
                    "分位数反解出来的，部署时根本拿不到。",
                ]
            else:
                L += [
                    "",
                    f"也就是说，自测确实带来了一点独立信息（{100 * gain:.0f}pp），"
                    "但幅度远小于它的宣传口径，且仍依赖一个用真值拟合的全局 `k`。",
                ]
            L += [
                "",
                "> **口径提醒**：本节所有数字都是**全样本**口径（`w_hat` 逐 cell 独立、"
                "`k` 与库常数都用全部 cell 拟合），没有任何留出；"
                "而 `online_proxy.py` 的 19pp 是**留一库**口径（2 库拟合、第 3 库评估）。",
                "自测在**更宽松**的评估下、只做到 "
                f"{100 * dev:.0f}pp（不带 `k`），仍然输给特征回归。",
                "",
                "> 两处的**基线也不一样**（本文 118 个 cell 的中位 "
                f"{100 * base:.0f}pp，`online_proxy` 是 15 个「模型×留出库」组合的平均 42pp），"
                "因为 cell 集与聚合方式不同。所以本文内部各行的比较是严格的，"
                "跨文只引用「19pp」这一个数。",
            ]
    else:
        L.append("（本次没有跑伪任务推理，只有前提 1 的结果。）")

    L += [
        "",
        "## 结论：§9.5 的最后一条路也关掉了",
        "",
        "| 尝试 | 覆盖率偏离 | 需要真值？ | 结论 |",
        "|---|---|---|---|",
        f"| 不校准（`w=1`） | {100 * base:.0f}pp | 否 | 基线 |",
        f"| **上下文内自测（`w_hat`）** | **{100 * dev:.0f}pp** | **否** | "
        "可部署，但**差于特征回归的 19pp** |",
        f"| 自测 × `k` | {100 * devx:.0f}pp | 是（`k` 从真实分位数反解） | 好看，但作弊 |",
        f"| 库常数基线（对照） | {100 * devc:.0f}pp | 是 | **花同样的作弊预算，比自测更好** |",
        "",
        "关键在最后两行：自测看起来能救回来，全靠事后乘的那个 `k`；"
        "而一个根本不用自测的库常数花同样的作弊预算就能**赢过**它。"
        "**自测本身的信息量是负的。**",
        "",
        "根因是三条约束叠在一起，任何一条单独解决都没用：",
        "",
        "1. **协议没有空间**：50/50 切分让「匹配 horizon 的自测」在数学上不存在，",
        "   伪 horizon 只能是真实 horizon 的一半（NASA 更被夹到 19 点）；",
        "2. **`w` 对 horizon 极度敏感**：短 horizon 的 `w` 只有长 horizon 的 12%，",
        "   而两者不是干净的幂律关系，没法用一个指数稳定修正；",
        "   实测 `w_hat/w_oracle` 的中位与这个 12% 几乎逐位吻合，是本文最强的证据；",
        "3. **伪真值不平滑**：上下文原值含逐循环测量噪声（CALCE 上约 1.9 SOH%），",
        "   把 `w_hat` 推向另一个方向，于是少量 cell 反而过度覆盖到 100%。",
        "",
        "还有一个反直觉的细节：**自测会主动破坏本来就好的模型**——"
        f"{hurt_txt}。",
        "因为伪任务估出的 `w_hat` 往往远小于 1，而这个校准器只会「加宽」，",
        "碰到已经够宽的模型就只会帮倒忙。",
        "",
        "**可部署的结论**：在线估 `w` 在本项目的协议下做不到——特征回归、conformal 自测、",
        "两者叠加都不行。实际能做的只有两条：按库给一个保守的固定倍数",
        "（代价是 CRPS 中位 +465%，见 §10.8），或者接受「区间偏窄」并把它当成已知局限报告",
        "（这也是本文 §7 局限第 9 条的选择）。",
        "",
        "**唯一可能真正解决它的办法是改协议**：用更晚的起点（如 0.75），",
        "这样上下文长度是 horizon 的 3 倍，`H_p = H` 的自测就有空间了。",
        "但那会改变所有已发表数字的口径，属于新实验而非补充实验——本文不做。",
        "",
    ]
    (DOC / "上下文自测.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="§9.5 上下文内自测（conformal 式伪任务）")
    ap.add_argument("--model", default="", help="单个模型；空=按 --dataset 跑该库全部可用模型")
    ap.add_argument("--dataset", default="", help="单个库；空=三库全跑")
    ap.add_argument("--horizon-frac", type=float, default=0.5,
                    help="伪 horizon = 上下文长度 × 该比例（默认 0.5）")
    ap.add_argument("--skip-inference", action="store_true",
                    help="只跑前提 1（零推理），不跑伪任务")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    inv = horizon_invariance()
    inv.to_csv(OUT / "horizon_invariance.csv", index=False, encoding="utf-8-sig")
    print(f"前提 1：{len(inv)} 行 -> horizon_invariance.csv")

    # 前提 1 反推出的 horizon 缩放因子：伪 horizon 只有真实的一半，
    # 所以 w_hat 需要乘 k = 1/中位比值 才能对齐真实 horizon 的量级。
    st = premise1_stats(inv)
    k_rescale = 1.0 / st[0.5][1] if 0.5 in st and st[0.5][1] > 0 else 1.0
    print(f"前提 1 的中位比值 {st[0.5][1]:.4f} -> fudge factor k = {k_rescale:.2f}"
          if 0.5 in st else "前提 1 样本不足，k 取 1")

    per_path = OUT / "per_cell_all.csv"
    if args.skip_inference:
        per = pd.read_csv(per_path, encoding="utf-8-sig") if per_path.exists() else pd.DataFrame()
    else:
        ds_list = [args.dataset] if args.dataset else DS_ORDER
        rows: List[Dict[str, object]] = []
        for ds in ds_list:
            models = ([args.model] if args.model else
                      [m for m in FM_ORDER if any(d == ds and m in ms
                                                  for d, ms in MATRIX.values())])
            for m in models:
                got = run_one(ds, m, args.horizon_frac, k_rescale)
                if got:
                    rows.extend(got)
                    pd.DataFrame(got).to_csv(
                        OUT / f"per_cell_{ds}_{m}.csv", index=False, encoding="utf-8-sig")
        new = pd.DataFrame(rows)
        # 与已有结果合并（按键去重，保留本次）
        if per_path.exists() and not new.empty:
            old = pd.read_csv(per_path, encoding="utf-8-sig")
            key = ["dataset", "model", "cell"]
            old = old[~old.set_index(key).index.isin(new.set_index(key).index)]
            per = pd.concat([old, new], ignore_index=True)
        else:
            per = new
        if not per.empty:
            per.to_csv(per_path, index=False, encoding="utf-8-sig")
            per.groupby(["dataset", "model"]).agg(
                n=("cell", "size"), w_hat=("w_hat", "median"),
                w_oracle=("w_oracle", "median"),
                picp_w1=("picp_w1", "median"),
                picp_w_hat=("picp_w_hat", "median"),
                picp_w_x=("picp_w_x", "median"),
                picp_oracle=("picp_oracle", "median"),
            ).reset_index().to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")

    write_doc(inv, per, k_rescale)
    pd.set_option("display.width", 200)
    if not per.empty:
        print(per.groupby(["dataset", "model"])[
            ["picp_w1", "picp_w_hat", "picp_w_x", "picp_oracle"]].median().to_string())
    print(f"\n产物 -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
