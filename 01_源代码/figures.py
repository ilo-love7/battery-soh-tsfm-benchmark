"""从 03_实验结果 画论文图（英文标注，避免中文字体缺失成方块）。

用法：
    python figures.py --run mit-zero-all --dataset mit
    python figures.py --run mit-zero-chronos-bolt --dataset mit

产出（04_文档报告/figures/）：
    fig1_trajectories.png   6 只代表性电芯的预测轨迹（上下文/真值/各模型）
    fig2_mae_box.png        逐 cell MAE 分布箱线图
    fig3_eol.png            EOL 命中率 + 预测 EOL 散点（无命中时退化为命中率条形图）
    fig4_sensitivity.png    起点敏感性（MAE vs start_frac，需先跑多起点结果）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT.parent / "03_实验结果"
FIG_DIR = ROOT.parent / "04_文档报告" / "figures"

COLORS = {
    "chronos-bolt": "#1f77b4", "chronos": "#17becf", "timesfm": "#d62728",
    "moirai": "#9467bd", "lag-llama": "#8c564b",
    "naive_linear": "#7f7f7f", "naive_last": "#bcbd22", "exp_decay": "#ff7f0e",
    "arima": "#2ca02c",
}

# 探测 boxplot 的刻度标签参数名（matplotlib 3.9+ 改名）
try:
    import inspect
    _BOXPLOT_KW = inspect.signature(plt.Axes.boxplot).parameters
except Exception:  # pragma: no cover
    _BOXPLOT_KW = {}

# fig3 右图用：哪些库的 EOL 真值可用（MIT 在 EOL 处截断，命中率恒 0，故不放）
# 值可以是单个目录名，也可以是列表（同一库分多次跑出的模型，如 moirai/lag-llama 走独立环境）
EOL_RUNS = {
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
# fig4 只从这些目录取数（多起点结果），不扫全库以免混入单点。
# ⚠️ 必须按数据集分键：之前是一个扁平列表，导致三个库画出的是同一张 MIT 曲线。
SENS_RUNS = {
    "mit_": ["mit-zero-chronos-bolt,naive_linear,exp_decay-sens"],
    "nasa_": ["nasa-zero-chronos-bolt,chronos,naive_linear,exp_decay-sens"],
    "calce_": ["calce-zero-chronos-bolt,chronos,naive_linear,exp_decay-sens"],
}


def load_run(run_dir: Path) -> Dict[str, dict]:
    """返回 {model: {'preds': {key: [pred, truth]}, 'ctx': {key: context}, 'per_cell': DataFrame}}"""
    out = {}
    for f in sorted(run_dir.glob("predictions_*.npz")):
        model = f.stem.replace("predictions_", "")
        with np.load(f) as z:
            preds = {k: z[k] for k in z.files if "|ctx" not in k}
            ctx = {k.split("|")[0]: z[k] for k in z.files if k.endswith("|ctx")}
        csv = run_dir / f"per_cell_{model}.csv"
        out[model] = {"preds": preds, "ctx": ctx,
                      "per_cell": pd.read_csv(csv) if csv.exists() else None}
    return out


def _split_key(k: str):
    cell, frac = k.rsplit("@", 1)
    return cell, float(frac)


def pick_cells(run: Dict[str, dict], frac: float, n: int = 6) -> List[str]:
    """按真值寿命分档挑 cell（短/中/长各 2 个），保证图有代表性。"""
    first = next(iter(run.values()))
    keys = [k for k in first["preds"] if abs(_split_key(k)[1] - frac) < 1e-9]
    lens = {k: len(first["preds"][k][0]) for k in keys}
    order = sorted(lens, key=lambda k: lens[k])
    if len(order) <= n:
        return order
    idx = np.linspace(0, len(order) - 1, n).round().astype(int)
    return [order[i] for i in idx]


def fig1(run: Dict[str, dict], frac: float, path: Path):
    cells = pick_cells(run, frac)
    ncol = 3
    nrow = int(np.ceil(len(cells) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow), squeeze=False)

    for ax, key in zip(axes.flat, cells):
        any_model = next(iter(run.values()))
        arr = any_model["preds"][key]
        n = len(arr[0])
        truth = arr[1]
        ctx = any_model.get("ctx", {}).get(key)
        if ctx is None:      # 老结果没有 ctx：按起点比例反推
            ctx_len = int(round(n * frac / (1 - frac)))
            ctx = np.full(ctx_len, truth[0])
        ctx_len = len(ctx)
        x_ctx = np.arange(ctx_len)
        x_fc = np.arange(ctx_len, ctx_len + n)

        ax.plot(x_ctx, ctx, color="0.55", lw=1.0, label="context (given)")
        ax.axvline(ctx_len - 0.5, color="0.7", ls="--", lw=0.8)
        ax.plot(x_fc, truth, color="k", lw=1.8, label="truth")
        for model, d in run.items():
            if key not in d["preds"]:
                continue
            p = d["preds"][key][0]
            ax.plot(x_fc, p, lw=1.1, alpha=0.9,
                    color=COLORS.get(model, None), label=model)
        ax.set_title(key, fontsize=9)
        ax.set_xlabel("cycle index", fontsize=8)
        ax.set_ylabel("SOH (%)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)

    for ax in list(axes.flat)[len(cells):]:
        ax.axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 5), fontsize=8, frameon=False)
    fig.suptitle(f"Zero-shot SOH trajectory forecast (start at {frac:.0%} of series)", fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig2(run: Dict[str, dict], frac: float, path: Path):
    data, labels = [], []
    for model, d in run.items():
        pc = d["per_cell"]
        if pc is None:
            continue
        sub = pc[(pc["start_frac"] - frac).abs() < 1e-9].dropna(subset=["mae_smooth"])
        if sub.empty:
            continue
        data.append(sub["mae_smooth"].to_numpy())
        labels.append(f"{model}\n(n={len(sub)})")
    if not data:
        return
    order = np.argsort([np.median(d) for d in data])
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(data)), 3.6))
    # matplotlib >=3.9 用 tick_labels，旧版用 labels
    label_kw = "tick_labels" if "tick_labels" in _BOXPLOT_KW else "labels"
    bp = ax.boxplot([data[i] for i in order], patch_artist=True, widths=0.6,
                    showfliers=True,
                    flierprops=dict(marker="o", ms=3, alpha=0.5),
                    **{label_kw: [labels[i] for i in order]})
    for patch, i in zip(bp["boxes"], order):
        model = labels[i].split("\n")[0]
        patch.set_facecolor(COLORS.get(model, "#cccccc"))
        patch.set_alpha(0.55)
    ax.set_ylabel("MAE vs smoothed truth (SOH %)", fontsize=9)
    ax.set_title(f"Per-cell error distribution (start at {frac:.0%})", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig3(run: Dict[str, dict], frac: float, path: Path,
         eol_runs: Dict[str, str] | None = None, threshold: float = 80.0):
    """左：各模型预测能压到的最低 SOH（说明为何 MIT 命中率为 0）；
    右：NASA / CALCE 的 EOL 命中率（真值可用，命中率非 0）。"""
    # --- 左：最低 SOH（用当前 run 的预测） ---
    lows, labels = [], []
    for model, d in run.items():
        pc = d["per_cell"]
        if pc is None:
            continue
        sub = pc[(pc["start_frac"] - frac).abs() < 1e-9]
        if sub.empty:
            continue
        vals = []
        for k, arr in d["preds"].items():
            if abs(_split_key(k)[1] - frac) > 1e-9:
                continue
            vals.append(float(np.min(arr[0])))
        if vals:
            lows.append(vals)
            labels.append(model)
    if not lows:
        return

    # --- 右：跨库 EOL 命中率 ---
    hits: Dict[str, Dict[str, float]] = {}
    if eol_runs:
        for ds, dirnames in eol_runs.items():
            if isinstance(dirnames, str):
                dirnames = [dirnames]
            for dirname in dirnames:
                rd = RESULTS_ROOT / dirname
                if not rd.exists():
                    print(f"[fig3] 跳过不存在的目录: {dirname}")
                    continue
                for model, d in load_run(rd).items():
                    pc = d["per_cell"]
                    if pc is None or "eol_hit" not in pc:
                        continue
                    sub = pc[(pc["start_frac"] - frac).abs() < 1e-9]
                    if sub.empty:
                        continue
                    hits.setdefault(ds, {})[model] = float(sub["eol_hit"].mean())

    ncol = 2 if hits else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5.0 * ncol, 3.6), squeeze=False)
    ax = axes.flat[0]
    order = np.argsort([np.median(v) for v in lows])
    label_kw = "tick_labels" if "tick_labels" in _BOXPLOT_KW else "labels"
    bp = ax.boxplot([lows[i] for i in order], patch_artist=True, widths=0.6,
                    showfliers=True, flierprops=dict(marker="o", ms=3, alpha=0.5),
                    **{label_kw: [labels[i] for i in order]})
    for patch, i in zip(bp["boxes"], order):
        patch.set_facecolor(COLORS.get(labels[i], "#cccccc"))
        patch.set_alpha(0.55)
    ax.axhline(threshold, color="crimson", ls="--", lw=1.2)
    ax.text(0.99, threshold, "EOL = 80%", color="crimson", fontsize=8,
            ha="right", va="bottom", transform=ax.get_yaxis_transform())
    ax.set_ylabel("lowest SOH reached by forecast (%)", fontsize=9)
    ax.set_title("How far each forecast descends (start at %.0f%%)" % (frac * 100), fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(labelsize=8, labelrotation=25)
    for lb in ax.get_xticklabels():
        lb.set_ha("right")

    if hits:
        ax2 = axes.flat[1]
        dss = list(hits)
        models = [m for m in COLORS if any(m in hits[d] for d in dss)]
        models = [m for m in models if any(m in hits[d] for d in dss)]
        w = 0.8 / max(len(dss), 1)
        xs = np.arange(len(models))
        prev = [np.nan] * len(models)     # 上一根柱的值，用于错开等高标签
        for j, ds in enumerate(dss):
            vals = [hits[ds].get(m, np.nan) for m in models]
            ax2.bar(xs + j * w, vals, width=w, label=ds.upper(), alpha=0.85)
            for i, (x, v) in enumerate(zip(xs + j * w, vals)):
                if np.isnan(v):
                    continue
                y = v + 0.02
                if not np.isnan(prev[i]) and abs(prev[i] - v) < 0.05:
                    y += 0.085          # 相邻柱等高：标签上移一行，避免叠字
                ax2.text(x, y, f"{v:.0%}", ha="center", fontsize=7)
                prev[i] = v
        ax2.set_xticks(xs + w * (len(dss) - 1) / 2)
        ax2.set_xticklabels(models, fontsize=8, rotation=25, ha="right")
        ax2.set_ylim(0, 1.12)
        ax2.set_ylabel("EOL hit rate", fontsize=9)
        ax2.set_title("EOL hit rate where truth is defined", fontsize=10)
        ax2.legend(fontsize=8, frameon=False, ncol=len(dss))
        ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fig4(run: Dict[str, dict], fracs_all: Dict[str, Dict[float, float]], path: Path):
    """敏感性：只画在**全部**起点上都有结果的模型，避免半截线/孤立点误导。"""
    n_frac = len(next(iter(fracs_all.values()))) if fracs_all else 0
    models = [m for m, d in fracs_all.items() if len(d) == n_frac and n_frac >= 2]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for m in sorted(models):
        g = sorted(fracs_all[m].items())
        ax.plot([x for x, _ in g], [y for _, y in g], marker="o", label=m,
                color=COLORS.get(m, None))
    ax.set_xlabel("forecast start (fraction of series)", fontsize=9)
    ax.set_ylabel("median MAE (SOH %)", fontsize=9)
    ax.set_title("Sensitivity to forecast start", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def collect_sensitivity(run_dirs: List[Path]) -> Dict[str, Dict[float, float]]:
    """只从**指定的**结果目录里收集 (model, frac) -> 中位 MAE，避免混入别处的 0.5。"""
    out: Dict[str, Dict[float, float]] = {}
    for rd in run_dirs:
        if not rd.exists():
            continue
        sj = rd / "summary.json"
        if not sj.exists():
            continue
        d = json.loads(sj.read_text(encoding="utf-8"))
        for key, summ in (d.get("summary") or {}).items():
            model, _, frac = key.rpartition("@")
            try:
                f = float(frac)
            except ValueError:
                continue
            mae = (summ.get("mae_smooth") or {}).get("median", np.nan)
            if mae is None or (isinstance(mae, float) and np.isnan(mae)):
                continue
            out.setdefault(model, {})[f] = float(mae)
    return out


def load_runs(dirs: List[Path]) -> Dict[str, dict]:
    """合并多个结果目录的模型（不同模型可能在不同虚拟环境里跑出来）。"""
    out: Dict[str, dict] = {}
    origin: Dict[str, str] = {}
    for d in dirs:
        if not d.exists():
            print(f"  !! 目录不存在，跳过：{d.name}")
            continue
        for k, v in load_run(d).items():
            if k in out:
                print(f"  !! {k} 在多个目录出现，保留先出现的（{origin[k]}），忽略 {d.name}")
                continue
            out[k] = v
            origin[k] = d.name
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="结果目录名，如 mit-zero-chronos-bolt")
    ap.add_argument("--runs", default=None,
                    help="分号分隔多个目录名，模型合并（如跨虚拟环境跑出的 moirai）。"
                         "注意结果目录名本身含逗号，故不能用逗号分隔")
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--prefix", default="", help="文件名前缀，如 mit_ / calce_（多数据集防覆盖）")
    args = ap.parse_args()

    if args.runs:
        dirs = [RESULTS_ROOT / s.strip() for s in args.runs.split(";") if s.strip()]
    elif args.run:
        dirs = [RESULTS_ROOT / args.run]
    else:
        raise SystemExit("必须给 --run 或 --runs")
    for d in dirs:
        if not d.exists():
            raise SystemExit(f"目录不存在: {d}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    run = load_runs(dirs)
    if not run:
        raise SystemExit("没有 predictions_*.npz")
    print("模型:", list(run))

    pre = args.prefix
    fig1(run, args.frac, FIG_DIR / f"{pre}fig1_trajectories.png")
    fig2(run, args.frac, FIG_DIR / f"{pre}fig2_mae_box.png")
    fig3(run, args.frac, FIG_DIR / f"{pre}fig3_eol.png", eol_runs=EOL_RUNS)
    sens_dirs = SENS_RUNS.get(pre, [])
    if sens_dirs:
        sens = collect_sensitivity([RESULTS_ROOT / d for d in sens_dirs])
        fig4(run, sens, FIG_DIR / f"{pre}fig4_sensitivity.png")
    else:
        print(f"[fig4] {pre or '(无前缀)'} 没有多起点敏感性结果，跳过（不生成假图）")
    print("图已输出到", FIG_DIR)


if __name__ == "__main__":
    main()
