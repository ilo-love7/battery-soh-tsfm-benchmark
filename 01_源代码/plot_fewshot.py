"""plot_fewshot.py —— 少样本微调曲线（MAE vs 支持集电池数 k）。

数据来源全部是真实跑出来的 summary.json，没有手工填数：
    k=0   mit-zero-chronos-bolt,chronos,timesfm,naive_linear-ctx512  （零样本，512 上下文）
    k=1   mit-zero-chronos-bolt-ft1
    k=4   mit-zero-chronos-bolt-ft4
    k=8   mit-zero-chronos-bolt-ft8
    k=19  mit-zero-chronos-bolt-ftall                                （batch1 里够长的全部）

同时画两条水平参考线：Chronos-T5 零样本（3.024）与 naive_linear（4.066）。

用法：python plot_fewshot.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "03_实验结果"
FIG_DIR = Path(__file__).resolve().parent.parent / "04_文档报告" / "figures"

# (k, 目录名, 模型 key, 标签)
# 微调评测放在 fewshot_eval/ 子目录：report.py 只扫一级目录，
# 这样微调结果不会覆盖主表里的零样本 chronos-bolt 行。
RUNS = [
    (0,  "mit-zero-chronos-bolt,chronos,timesfm,naive_linear-ctx512", "chronos-bolt", "zero-shot"),
    (1,  "fewshot_eval/mit-zero-chronos-bolt-ft1",   "chronos-bolt", "ft-1"),
    (4,  "fewshot_eval/mit-zero-chronos-bolt-ft4",   "chronos-bolt", "ft-4"),
    (8,  "fewshot_eval/mit-zero-chronos-bolt-ft8",   "chronos-bolt", "ft-8"),
    (19, "fewshot_eval/mit-zero-chronos-bolt-ftall", "chronos-bolt", "ft-19"),
]


def read_mae(run: str, model: str, frac: float = 0.5):
    p = RESULTS_ROOT / run / "summary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    for key, v in (d.get("summary") or {}).items():
        m, _, f = key.rpartition("@")
        if m == model and abs(float(f) - frac) < 1e-9:
            return (float(v.get("mae_smooth", {}).get("median", float("nan"))),
                    float(v.get("eol_hit", {}).get("mean", float("nan"))))
    return None


def main():
    ks, maes, hits = [], [], []
    for k, run, model, _ in RUNS:
        r = read_mae(run, model)
        if r is None:
            print(f"  跳过 k={k}：{run} 缺失")
            continue
        ks.append(k)
        maes.append(r[0])
        hits.append(r[1])
        print(f"  k={k:2d}  MAE {r[0]:.3f}  EOL命中 {r[1]:.0%}")

    if not ks:
        raise SystemExit("没有可用数据")

    # 参考线
    refs = [("Chronos-T5 (zero-shot)", "chronos", "#17becf"),
            ("naive_linear", "naive_linear", "#7f7f7f")]
    ref_vals = []
    for label, model, color in refs:
        r = read_mae("mit-zero-chronos-bolt,chronos,timesfm,naive_linear-ctx512", model)
        if r:
            ref_vals.append((label, r[0], color))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.4))

    ax.plot(ks, maes, "o-", color="#1f77b4", lw=1.8, ms=6, label="Chronos-Bolt + fine-tune")
    for k, v in zip(ks, maes):
        # 首点标在右下、末点标在左上，避免压住参考线与坐标轴
        off = (8, -3) if k == min(ks) else ((-2, 9) if k == max(ks) else (0, 9))
        ha = "left" if k == min(ks) else ("right" if k == max(ks) else "center")
        ax.annotate(f"{v:.2f}", (k, v), fontsize=7.5, xytext=off,
                    textcoords="offset points", ha=ha, color="#1f77b4")
    for label, v, color in ref_vals:
        ax.axhline(v, ls="--", lw=1, color=color, alpha=0.85)
        ax.annotate(label, (max(ks) + 1.6, v), fontsize=7, color=color,
                    xytext=(0, 3), textcoords="offset points", ha="right")
    ax.set_xlabel("number of support cells (MIT batch1)", fontsize=9)
    ax.set_ylabel("median MAE (SOH %)", fontsize=9)
    ax.set_title("Few-shot fine-tuning closes most of the gap", fontsize=10)
    ax.set_xticks(ks)
    ax.set_xlim(min(ks) - 1.2, max(ks) + 1.8)
    ax.grid(alpha=0.3)

    ax2.plot(ks, [h * 100 for h in hits], "s-", color="#d62728", lw=1.6, ms=5)
    ax2.set_xlabel("number of support cells (MIT batch1)", fontsize=9)
    ax2.set_ylabel("EOL hit rate (%)", fontsize=9)
    ax2.set_title("EOL hit rate vs support size", fontsize=10)
    ax2.set_xticks(ks)
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig6_fewshot.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print("图已输出", out)


if __name__ == "__main__":
    main()
