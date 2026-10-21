"""机制分析：模型"敢外推多少衰减"与误差的关系。

核心假设：三库的排序差异不是"模型强弱"，而是**预测的衰减速率是否匹配真值**。
定义两个量（都在每个 cell 上算，再取中位数）：

  true_ratio  = 预测窗内真值降幅 / 上下文段降幅   （数据侧，与模型无关）
  pred_ratio  = 预测窗内预测降幅 / 预测窗内真值降幅 （模型侧）

即：true_ratio 描述"这段数据本身未来比过去掉得快几倍"，
pred_ratio 描述"模型画出来的降幅是真值的几倍"。
两者都接近 1 时模型把衰减速率外推对了，MAE 应该小。
用 |log(pred_ratio / true_ratio)| 作为"速率失配度"量化，检查它与 MAE 的秩相关。

⚠️ true_ratio 是**手工常量**（下方 TRUE_RATIO），由 diag_curvature.py 实测后填进来，
不是本脚本算的。换数据口径（清洗规则、起点、CALCE 筛选）后必须重跑
diag_curvature.py 并更新它，否则机制图整体失真。

输出：
  03_实验结果/汇总_衰减保留率.csv
  04_文档报告/figures/fig5_shrinkage.png

用法：python analyze_shrinkage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs.config import PROJECT_ROOT, RESULTS_ROOT  # noqa: E402

DOCS_DIR = PROJECT_ROOT / "04_文档报告"

# (数据集, 结果目录, 展示名)
RUNS = [
    ("mit", "mit-zero-chronos-bolt,chronos,timesfm,naive_linear-ctx512"),
    ("mit", "mit-zero-moirai"),
    ("mit", "mit-zero-lag-llama"),
    ("mit", "mit-zero-naive_last,naive_linear,exp_decay-refactor"),
    ("mit", "mit-zero-arima"),
    ("nasa", "nasa-zero-chronos-bolt,chronos,timesfm,naive_linear,naive_last,exp_decay"),
    ("nasa", "nasa-zero-moirai"),
    ("nasa", "nasa-zero-lag-llama"),
    ("calce", "calce-zero-chronos-bolt,chronos,timesfm,naive_linear,naive_last,exp_decay"),
    ("calce", "calce-zero-moirai"),
    ("calce", "calce-zero-lag-llama"),
]

# 数据侧 true_ratio（由 diag_curvature.py 实测，见执行记录）
# ⚠️ 换数据口径后必须重跑 diag_curvature.py 并更新这里：
#    CALCE 第四轮（电压窗口）从 0.69 → 1.65，因为真值现在真的衰减到 40% 而不是停在 84%。
TRUE_RATIO = {"mit": 8.88, "calce": 1.65, "nasa": 0.65}


def collect():
    recs = []
    for ds, run in RUNS:
        d = RESULTS_ROOT / run
        if not d.exists():
            continue
        for f in sorted(d.glob("predictions_*.npz")):
            model = f.stem.replace("predictions_", "")
            z = np.load(f)
            rows = []
            for k in z.files:
                if k.endswith("|ctx"):
                    continue
                a = z[k]
                if a.ndim != 2 or a.shape[0] < 2:
                    continue
                p, t = a[0], a[1]
                dt = float(t[0] - t[-1])
                dp = float(p[0] - p[-1])
                rows.append({
                    "bias": float(np.mean(p - t)),
                    "mae": float(np.mean(np.abs(p - t))),
                    "ratio": dp / dt if abs(dt) > 1e-6 else np.nan,
                })
            if not rows:
                continue
            r = pd.DataFrame(rows)
            recs.append({
                "dataset": ds, "model": model, "n": len(r),
                "mae": r.mae.median(), "bias": r.bias.median(),
                "pred_ratio": r.ratio.median(),
            })
    df = pd.DataFrame(recs)
    # 同一 (数据集, 模型) 可能出现在多个 run 里（如 ctx512 与全基线 run）。
    # RUNS 已按优先级排列（公平对照在前），保留第一次出现的即可。
    df = df.drop_duplicates(subset=["dataset", "model"], keep="first")
    df["true_ratio"] = df.dataset.map(TRUE_RATIO)
    # 速率失配度：负值代表预测的衰减方向都反了，取 log 前先夹到正值
    eps = 1e-3
    df["mismatch"] = np.abs(np.log(df.pred_ratio.clip(lower=eps) / df.true_ratio))
    return df


def rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman 秩相关（不依赖 scipy）。"""
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den else np.nan


def plot(df: pd.DataFrame, path: Path):
    """三联图：每个数据集一个面板（各自 true_ratio 不同，不能混在一个坐标系里拟合）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dss = [d for d in ["mit", "nasa", "calce"] if (df.dataset == d).any()]
    fig, axes = plt.subplots(1, len(dss), figsize=(4.1 * len(dss), 3.5), squeeze=False)
    colors = {"mit": "#1f77b4", "nasa": "#ff7f0e", "calce": "#2ca02c"}

    for ax, ds in zip(axes[0], dss):
        g = df[df.dataset == ds]
        # Lag-Llama 单独标记：它是唯一非 512 上下文的模型，且预测路径噪声极大，
        # 会出现"终点降幅对得上、路径全错"的情况，破坏失配度-误差的单调关系。
        g_ok = g[g.model != "lag-llama"]
        g_ll = g[g.model == "lag-llama"]
        ax.scatter(g_ok.mismatch, g_ok.mae, s=52, alpha=0.9, color=colors.get(ds, "#888"),
                   edgecolors="white", linewidths=0.7)
        if len(g_ll):
            ax.scatter(g_ll.mismatch, g_ll.mae, s=58, facecolors="none",
                       edgecolors=colors.get(ds, "#888"), linewidths=1.4, marker="s")
        for j, (_, r) in enumerate(g.iterrows()):
            # 交替偏移，避免相邻点标签互相压住
            dy = 5 if j % 2 == 0 else -9
            ax.annotate(r.model, (r.mismatch, r.mae), fontsize=6.0,
                        xytext=(3, dy), textcoords="offset points", alpha=0.85)
        if len(g_ok) >= 3:
            k, b = np.polyfit(g_ok.mismatch, g_ok.mae, 1)
            xs = np.linspace(g_ok.mismatch.min(), g_ok.mismatch.max(), 50)
            ax.plot(xs, k * xs + b, "--", color="#555", lw=1, alpha=0.75)
            rho = rank_corr(g_ok.mismatch.to_numpy(), g_ok.mae.to_numpy())
            rho_all = rank_corr(g.mismatch.to_numpy(), g.mae.to_numpy())
            ax.set_title(f"{ds.upper()}  (true ratio {TRUE_RATIO[ds]:.2f})\n"
                         f"ρ={rho:+.2f} (n={len(g_ok)})   ρ={rho_all:+.2f} 含 lag-llama",
                         fontsize=8.5)
        ax.set_xlabel("fade-rate mismatch  |log(pred / true)|", fontsize=8)
        if ax is axes[0][0]:
            ax.set_ylabel("median MAE (SOH %)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Forecast error is explained by how much fade the model dares to extrapolate",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    df = collect()
    if df.empty:
        raise SystemExit("没有找到结果")
    df = df.sort_values(["dataset", "mae"])
    out_csv = RESULTS_ROOT / "汇总_衰减保留率.csv"
    df.round(4).to_csv(out_csv, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    for ds, g in df.groupby("dataset"):
        print(f"=== {ds}（true_ratio={TRUE_RATIO[ds]}）===")
        print(g[["model", "n", "mae", "pred_ratio", "mismatch"]]
              .round(3).to_string(index=False))
        print()

    print("--- 秩相关（mismatch vs MAE）---")
    for ds, g in df.groupby("dataset"):
        if len(g) >= 3:
            g_ok = g[g.model != "lag-llama"]
            rho = rank_corr(g_ok.mismatch.to_numpy(), g_ok.mae.to_numpy())
            rho_all = rank_corr(g.mismatch.to_numpy(), g.mae.to_numpy())
            print(f"  {ds:6}: ρ={rho:+.2f} (n={len(g_ok)}, 不含 lag-llama)   "
                  f"ρ={rho_all:+.2f} (n={len(g)}, 全部)")
    print("  注 1：不同库的 true_ratio 差一个量级（8.88 vs 1.65 vs 0.65），合并拟合无意义，只报分库。")
    print("  注 2：Lag-Llama 是唯一非 512 上下文（原生 32）的模型，且预测路径噪声极大，")
    print("        会出现「总降幅对得上、路径全错」的情况，单独列出以说明机制解释的边界。")

    fig_dir = DOCS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot(df, fig_dir / "fig5_shrinkage.png")
    print("已保存", out_csv)
    print("已保存", fig_dir / "fig5_shrinkage.png")


if __name__ == "__main__":
    main()
