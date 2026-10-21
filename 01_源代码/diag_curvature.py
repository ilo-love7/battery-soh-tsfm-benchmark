"""诊断：各数据集真值在"预测窗"内的形状（线性度 / 曲率 / 斜率）。

目的：解释为什么 naive_linear 在 CALCE 上最好、在 MIT 上却输给 Chronos-T5。
做法：复用 data/protocol.relative_split（与评测完全同口径），取 truth_smooth，
对其做一次线性拟合，报告
  R2_lin   —— 线性拟合的解释度（越接近 1 越"直"）
  curv     —— 二阶差分的中位绝对值（SOH%/循环^2，越大越"弯"）
  slope    —— 线性斜率（SOH%/循环，越负衰减越快）
  accel    —— 后半段斜率 / 前半段斜率（>1 表示衰减在加速，即 knee）
  slope_ctx/hor —— 上下文段与预测段的线性斜率（SOH%/循环）
  knee_ratio    —— hor/ctx 斜率比。>1 表示"未来比过去掉得更快"，
                   即上下文几乎不含未来衰减信息（线性外推必然低估）。

用法：python diag_curvature.py --dataset mit --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs.config import DATASETS, PROTOCOL  # noqa: E402
from data.loaders import load_dataset  # noqa: E402
from data.protocol import clean_series, relative_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--frac", type=float, default=0.5)
    args = ap.parse_args()

    cells, meta = load_dataset(args.dataset, DATASETS, split=args.split, with_meta=True)
    rows = []
    for cid, df in cells.items():
        d = clean_series(df, PROTOCOL["outlier_thresh"])
        cell_eol = (meta.get(cid) or {}).get("cycle_life")
        sp = relative_split(
            d, frac=args.frac,
            smooth_window=PROTOCOL["smooth_window"],
            smooth_order=PROTOCOL["smooth_order"],
            ref_points=PROTOCOL["soh_ref_points"],
            eol_true=cell_eol,
        )
        if sp is None:
            continue
        seg = np.asarray(sp.truth_smooth, dtype=float)
        if len(seg) < 8:
            continue
        ctx = np.asarray(sp.context, dtype=float)
        ctx_slope = (float(np.polyfit(np.arange(len(ctx)), ctx, 1)[0])
                     if len(ctx) >= 5 else np.nan)
        x = np.arange(len(seg), dtype=float)
        a, b = np.polyfit(x, seg, 1)
        pred = a * x + b
        ss_res = float(((seg - pred) ** 2).sum())
        ss_tot = float(((seg - seg.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        d2 = np.diff(seg, 2)
        half = len(seg) // 2
        s1 = float(np.polyfit(np.arange(half), seg[:half], 1)[0]) if half > 2 else np.nan
        s2 = float(np.polyfit(np.arange(len(seg) - half), seg[half:], 1)[0]) if half > 2 else np.nan
        rows.append({
            "cell_id": cid, "n_total": len(sp.context) + len(seg),
            "n_horizon": len(seg),
            "soh_start": float(seg[0]), "soh_end": float(seg[-1]),
            "R2_lin": r2,
            "curv": float(np.median(np.abs(d2))),
            "slope": float(a),
            "accel": s2 / s1 if (s1 and abs(s1) > 1e-9) else np.nan,
            "slope_ctx": ctx_slope,
            "knee_ratio": (a / ctx_slope) if abs(ctx_slope) > 1e-9 else np.nan,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("没有可用 cell")
    pd.set_option("display.width", 200)
    print(f"=== {args.dataset} / {args.split or 'all'} / start={args.frac} / n={len(df)} ===")
    print(df.round(4).to_string(index=False))
    print("\n--- 汇总（中位数）---")
    print(f"  R2_lin : {df.R2_lin.median():.4f}   (越接近 1 越线性)")
    print(f"  curv   : {df.curv.median():.4f}    (SOH%/循环^2)")
    print(f"  slope  : {df.slope.median():.4f}   (SOH%/循环)")
    print(f"  accel  : {df.accel.median():.3f}   (>1 = 后期衰减加速)")
    print(f"  slope_ctx : {df.slope_ctx.median():.4f}  slope_hor : {df.slope.median():.4f}")
    print(f"  knee_ratio: {df.knee_ratio.median():.2f}  (>1 = 未来比过去掉得更快，线性外推低估)")
    print(f"  SOH 区间: {df.soh_start.median():.2f} -> {df.soh_end.median():.2f}")

    out = (Path(__file__).resolve().parent / "data" / "processed"
           / f"curv_{args.dataset}_{args.split or 'all'}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("已保存", out)


if __name__ == "__main__":
    main()
