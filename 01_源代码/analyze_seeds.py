"""多种子方差分析：同一配置换随机种子，结论会不会变。

背景
----
Chronos-T5 / Moirai / Lag-Llama 都是采样型模型（采样 N 条再取 0.5 分位），
主表只跑了"默认种子"一次。审稿人会问：**换个种子排名还成立吗？**

本脚本读 `03_实验结果/_seeds/<库>-s<种子>/summary.json`（由
`run_benchmark.py --seed K --output-dir ...` 生成），与主表（默认种子）并列，
给出每个模型的种子间波动与"排名是否稳定"的判定。

用法
----
    python analyze_seeds.py

产物
----
    03_实验结果/种子方差.md / .csv
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "03_实验结果"
SEEDS = [1, 2, 3]

# 主表（默认种子）所在目录：与 figures.py / stats_significance.py 保持一致
MAIN: Dict[str, List[str]] = {
    "mit": [
        "mit-zero-chronos-bolt,chronos,timesfm,naive_linear-ctx512",
        "mit-zero-moirai",
        "mit-zero-lag-llama",
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

# 只统计采样型模型；确定性模型（Bolt / TimesFM / 基线）换种子不会有任何变化
SAMPLING = ["chronos", "moirai", "lag-llama"]
DISPLAY = {"chronos": "Chronos-T5", "moirai": "Moirai", "lag-llama": "Lag-Llama"}
FRAC = 0.5


def read_mae(path: Path) -> Dict[str, float]:
    """读 summary.json，返回 {model: 中位 MAE_smooth}（只取 FRAC 那个起点）。"""
    out: Dict[str, float] = {}
    if not path.exists():
        return out
    d = json.loads(path.read_text(encoding="utf-8"))
    for key, summ in (d.get("summary") or {}).items():
        model, _, frac = key.rpartition("@")
        try:
            if abs(float(frac) - FRAC) > 1e-9:
                continue
        except ValueError:
            continue
        mae = (summ.get("mae_smooth") or {}).get("median")
        if mae is not None:
            out.setdefault(model, float(mae))
    return out


def collect(dataset: str) -> pd.DataFrame:
    rows = []
    # 默认种子（主表）
    default: Dict[str, float] = {}
    for dirname in MAIN[dataset]:
        for m, v in read_mae(RESULTS / dirname / "summary.json").items():
            default.setdefault(m, v)

    for model in SAMPLING:
        vals = [default.get(model)]
        for s in SEEDS:
            v = read_mae(RESULTS / "_seeds" / f"{dataset}-s{s}" / "summary.json").get(model)
            vals.append(v)
        if all(v is None for v in vals):
            continue
        arr = np.array([v for v in vals if v is not None], dtype=float)
        rows.append({
            "dataset": dataset, "model": model, "display": DISPLAY.get(model, model),
            "default": vals[0],
            **{f"seed{s}": vals[i + 1] for i, s in enumerate(SEEDS)},
            "n_seeds": len(arr), "mean": float(arr.mean()), "std": float(arr.std(ddof=1))
            if len(arr) > 1 else float("nan"),
            "min": float(arr.min()), "max": float(arr.max()),
            "spread": float(arr.max() - arr.min()),
            "cv_pct": float(arr.std(ddof=1) / arr.mean() * 100) if len(arr) > 1 else float("nan"),
        })
    return pd.DataFrame(rows)


def rank_check(dataset: str, df: pd.DataFrame) -> str:
    """种子间的排名是否一致（用每个种子的值重排，看顺序是否与默认种子一致）。"""
    if df.empty:
        return ""
    # 基准排序：默认种子
    base = df.sort_values("default")["model"].tolist()
    flips = []
    for s in SEEDS:
        col = f"seed{s}"
        if df[col].isna().any():
            continue
        order = df.sort_values(col)["model"].tolist()
        if order != base:
            flips.append(f"seed{s}: {' < '.join(DISPLAY[m] for m in order)}")
    if not flips:
        return f"三个种子下排序与默认种子一致（{' < '.join(DISPLAY[m] for m in base)}）"
    return "排序发生翻转 → " + "；".join(flips)


def main():
    frames = [collect(ds) for ds in ["mit", "nasa", "calce"]]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if df.empty:
        raise SystemExit("没有 _seeds 结果，先跑 run_benchmark.py --seed K")

    df.to_csv(RESULTS / "种子方差.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 多种子方差分析（自动生成）",
        "",
        "> 由 `01_源代码/analyze_seeds.py` 生成，勿手改。",
        "> 只统计**采样型**模型（Chronos-T5 / Moirai / Lag-Llama）：Bolt、TimesFM 与四个基线是确定性的，",
        "> 换种子不会有任何变化，因此不列入。",
        "> `default` = 主表使用的默认种子（Chronos-T5 为 42，Moirai/Lag-Llama 为 0）；",
        "> `seed1..3` = `--seed K` 的复跑。指标为中位 MAE_smooth（SOH%），起点 50%。",
        "",
    ]
    for ds in ["mit", "nasa", "calce"]:
        sub = df[df.dataset == ds]
        if sub.empty:
            continue
        title = {"mit": "MIT-Stanford · test（43 支）", "nasa": "NASA PCoE（4 支）",
                 "calce": "CALCE UMD（14 支）"}[ds]
        lines += [f"## {title}", "",
                  "| 模型 | 默认种子 | seed1 | seed2 | seed3 | 均值 | 标准差 | 极差 | 变异系数 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            def fmt(x):
                return "—" if pd.isna(x) else f"{x:.3f}"
            lines.append(
                f"| {r['display']} | {fmt(r['default'])} | {fmt(r['seed1'])} | {fmt(r['seed2'])} | "
                f"{fmt(r['seed3'])} | {r['mean']:.3f} | {r['std']:.3f} | {r['spread']:.3f} | "
                f"{r['cv_pct']:.1f}% |")
        lines += ["", f"> {rank_check(ds, sub)}", ""]

    lines += [
        "## 读法",
        "",
        "- **变异系数（CV）** 是最该看的列：它把\"波动\"折算成相对量级。CV < 5% 说明该模型在该库上",
        "  的排名是稳的；CV > 10% 说明单次结果不足以支撑名次之争。",
        "- **极差** 与 §4.7 的 bootstrap CI 回答的是两个不同问题：CI 问\"换一批电池会怎样\"，",
        "  极差问\"换一个随机种子会怎样\"。两者都大，说明该结论需要更多数据才能定论。",
        "- NASA 只有 4 支电池，CV 天然偏大——这也是本文只把它当方向性证据的原因。",
    ]
    (RESULTS / "种子方差.md").write_text("\n".join(lines), encoding="utf-8")

    for ds in ["mit", "nasa", "calce"]:
        sub = df[df.dataset == ds]
        if sub.empty:
            continue
        print(f"\n=== {ds} ===")
        for _, r in sub.iterrows():
            print(f"{r['display']:>12s} default={r['default']:.3f} "
                  f"seeds={r['min']:.3f}-{r['max']:.3f} CV={r['cv_pct']:.1f}%")
        print("  ", rank_check(ds, sub))
    print(f"\n已保存 {RESULTS / '种子方差.md'}")


if __name__ == "__main__":
    main()
