"""统计显著性：cell 级 bootstrap 置信区间 + 与线性外推的配对检验。

为什么需要这个脚本
------------------
主表报的是"中位 MAE"，但 43 支 / 14 支 / 4 支电池本身是样本，
换一批电池数字就会变。原文多处用 1%–2% 的差距下判断（如 Bolt 4.110 vs 线性 4.066），
在 §7 里又自己承认"不应解读为显著"——等于把结论的可信度让出去了。

这里补两件事：
1. **bootstrap 95% CI**：对 cell 有放回重采样，重算中位 MAE，给出区间。
   看两个模型的 CI 是否重叠，就知道差距是否在噪声量级。
2. **配对 Wilcoxon 符号秩检验**：同一批 cell 上逐格比较 MAE，
   检验"这个模型比线性外推好/差"是否系统性成立。
   配对的前提是同一 cell —— 这比独立样本检验更贴合本实验设计。

用法
----
    python stats_significance.py                 # 三库全跑
    python stats_significance.py --dataset calce # 只跑一个库

产物
----
    03_实验结果/显著性检验.csv / .md
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "03_实验结果"

# 与 figures.py 保持一致的"权威目录"清单（同一模型可能在多个目录里，先出现的优先）
RUNS: Dict[str, List[str]] = {
    "mit": [
        "mit-zero-chronos-bolt,chronos,timesfm,naive_linear-ctx512",
        "mit-zero-moirai",
        "mit-zero-lag-llama",
        "mit-zero-naive_last,naive_linear,exp_decay-refactor",
        "mit-zero-arima",
    ],
    "nasa": [
        "nasa-zero-chronos-bolt,chronos,timesfm,naive_linear,naive_last,exp_decay",
        "nasa-zero-moirai",
        "nasa-zero-lag-llama",
    ],
    "calce": [
        "calce-zero-chronos-bolt,chronos,timesfm,naive_linear,naive_last,exp_decay",
        "calce-zero-moirai",
        "calce-zero-lag-llama",
    ],
}

FRAC = 0.5
N_BOOT = 10000
SEED = 0
BASELINE = "naive_linear"

DISPLAY = {
    "chronos": "Chronos-T5", "chronos-bolt": "Chronos-Bolt", "timesfm": "TimesFM",
    "moirai": "Moirai", "lag-llama": "Lag-Llama", "naive_linear": "naive_linear",
    "naive_last": "naive_last", "exp_decay": "exp_decay", "arima": "ARIMA",
}


def collect(dataset: str) -> Dict[str, pd.Series]:
    """返回 {模型: Series(index=cell_id, values=mae_smooth)}，只取 FRAC 那一个起点。"""
    out: Dict[str, pd.Series] = {}
    for dirname in RUNS[dataset]:
        rd = RESULTS_ROOT / dirname
        if not rd.exists():
            continue
        for f in sorted(rd.glob("per_cell_*.csv")):
            model = f.stem.replace("per_cell_", "")
            if model in out:
                continue
            df = pd.read_csv(f)
            if "error" in df.columns:
                df = df[df["error"].isna()]
            df = df[(df["start_frac"] - FRAC).abs() < 1e-9]
            if df.empty or "mae_smooth" not in df.columns:
                continue
            s = df.set_index("cell_id")["mae_smooth"].astype(float).dropna()
            if len(s):
                out[model] = s
    return out


def boot_ci(x: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    """中位数的 percentile bootstrap 置信区间。"""
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    meds = np.median(x[idx], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def paired_boot_diff(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
    """配对 bootstrap：中位差 (a-b) 的 95% CI。a、b 已按同一 cell 顺序对齐。"""
    rng = np.random.default_rng(seed)
    n = len(a)
    if n < 3:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    d = np.median(a[idx] - b[idx], axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.stats import wilcoxon as _w
    except ImportError:
        return float("nan")
    if len(a) < 6 or np.allclose(a, b):
        return float("nan")
    try:
        return float(_w(a, b, zero_method="wilcox").pvalue)
    except Exception:
        return float("nan")


def analyze(dataset: str) -> List[dict]:
    data = collect(dataset)
    if not data:
        return []
    base = data.get(BASELINE)
    rows = []
    for model, s in data.items():
        x = s.values
        lo, hi = boot_ci(x)
        row = {
            "dataset": dataset, "model": model, "display": DISPLAY.get(model, model),
            "n": len(x), "mae_median": float(np.median(x)),
            "ci_lo": lo, "ci_hi": hi,
            "d_vs_base": float("nan"), "d_lo": float("nan"), "d_hi": float("nan"),
            "p_wilcoxon": float("nan"), "sig": "",
        }
        if base is not None and model != BASELINE:
            common = s.index.intersection(base.index)
            a, b = s.loc[common].values, base.loc[common].values
            # 点估计必须与 CI 同源：都用"配对差的中位数"，
            # 不能用 median(a)-median(b)（那是另一个统计量，会落在自己的 CI 之外）
            row["d_vs_base"] = float(np.median(a - b))
            dlo, dhi = paired_boot_diff(a, b)
            row["d_lo"], row["d_hi"] = dlo, dhi
            p = wilcoxon(a, b)
            row["p_wilcoxon"] = p
            if not np.isnan(p):
                row["sig"] = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            # 差值 CI 不含 0 ⇒ 差距在 cell 层面稳定
            if not (np.isnan(dlo) or np.isnan(dhi)):
                row["ci_excludes_0"] = int(dlo > 0 or dhi < 0)
        rows.append(row)
    return sorted(rows, key=lambda r: r["mae_median"])


def to_markdown(all_rows: List[dict]) -> str:
    lines = [
        "# 统计显著性检验（自动生成）",
        "",
        "> 由 `01_源代码/stats_significance.py` 生成，勿手改。",
        "> 指标为 `mae_smooth`（平滑真值，SOH%），起点 50%。",
        "> **95% CI**：cell 级 percentile bootstrap（10000 次重采样）。",
        "> **Δ**：与 naive_linear 的**逐 cell 配对差的中位数**（负值=更好）。",
        "> ⚠️ 配对 Δ 与\"两个中位 MAE 之差\"不是同一个数：配对消除了 cell 难度差异，",
        "> 通常更小（例如 CALCE 上 Chronos-Bolt 的中位 MAE 差 7.31，配对 Δ 只有 4.77）。",
        "> **Δ 的 95% CI**：配对 bootstrap（同一批 cell 同步重采样）。",
        "> **p**：同一批 cell 上的配对 Wilcoxon 符号秩检验；`n.s.` 表示 p≥0.05。",
        "",
    ]
    for ds in ["mit", "nasa", "calce"]:
        rows = [r for r in all_rows if r["dataset"] == ds]
        if not rows:
            continue
        title = {"mit": "MIT-Stanford · test", "nasa": "NASA PCoE · all",
                 "calce": "CALCE UMD · all"}[ds]
        lines.append(f"## {title}（n={rows[0]['n']}）")
        lines.append("")
        lines.append("| 模型 | MAE 中位 | 95% CI | Δ vs naive_linear | Δ 的 95% CI | p | 显著性 |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in rows:
            ci = f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]" if not np.isnan(r["ci_lo"]) else "—"
            if np.isnan(r["d_vs_base"]):
                d, dci, p, sig = "—", "—", "—", "（基准）"
            else:
                d = f"{r['d_vs_base']:+.2f}"
                dci = (f"[{r['d_lo']:+.2f}, {r['d_hi']:+.2f}]"
                       if not np.isnan(r["d_lo"]) else "—")
                p = f"{r['p_wilcoxon']:.3f}" if not np.isnan(r["p_wilcoxon"]) else "—"
                sig = r["sig"] or "—"
            lines.append(f"| {r['display']} | {r['mae_median']:.3f} | {ci} | {d} | {dci} | {p} | {sig} |")
        lines.append("")
        if rows[0]["n"] < 6:
            lines.append(f"> n={rows[0]['n']}，样本量不足以做 Wilcoxon 符号秩检验，"
                         "p 与显著性留空；Δ 的 CI 也极宽，仅作方向性参考。")
            lines.append("")
    lines.append("> 读法：两个模型的 95% CI 明显重叠时，排名差异不宜当作结论；")
    lines.append("> `Δ 的 95% CI` 跨 0 或 p 为 `n.s.` 时，只能说\"方向一致\"，不能说\"显著更好\"。")
    lines.append("> ⚠️ **多重比较未校正**：每库都对同一个 naive_linear 做了 7–8 次检验，")
    lines.append("> 即使原假设全为真，也会有约 5% 的检验偶然显著。因此 p 只用于确认方向，")
    lines.append("> 排序一律以效应量（Δ 与其 CI）与 §5 的机制解释为准。")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, choices=["mit", "nasa", "calce"])
    args = ap.parse_args()

    datasets = [args.dataset] if args.dataset else ["mit", "nasa", "calce"]
    all_rows: List[dict] = []
    for ds in datasets:
        rows = analyze(ds)
        if not rows:
            print(f"!! {ds}: 没找到 per_cell 结果")
            continue
        all_rows.extend(rows)
        print(f"\n=== {ds} (n={rows[0]['n']}) ===")
        for r in rows:
            print(f"{r['display']:>14s} {r['mae_median']:7.3f} "
                  f"[{r['ci_lo']:6.2f},{r['ci_hi']:6.2f}] "
                  f"Δ{r['d_vs_base']:+7.2f} p={r['p_wilcoxon']:.3f} {r['sig']}")

    if not all_rows:
        raise SystemExit("没有结果")
    df = pd.DataFrame(all_rows)
    csv_path = RESULTS_ROOT / "显著性检验.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path = RESULTS_ROOT / "显著性检验.md"
    md_path.write_text(to_markdown(all_rows), encoding="utf-8")
    print(f"\n已保存 {csv_path}\n已保存 {md_path}")


if __name__ == "__main__":
    main()
