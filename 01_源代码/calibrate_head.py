"""拟合 2 参数单调校准头（执行记录 §9.3）。

    p' = c + s·(p − c) + b        （c = 上下文最后一个点，逐 chunk 施加）

在**冻结**的基础模型外面学 2 个参数，用"敢不敢外推"这一机制发现反推
更省的方案：如果 few-shot 微调的收益主要来自外推意愿，那么显式建模
外推意愿就应该能拿到大部分收益，而不必更新 190 MB 权重。

用法：
    python 01_源代码/calibrate_head.py --dataset mit --split train
    python 01_源代码/calibrate_head.py --dataset mit --split train --fast   # 冒烟

拟合边界（与 few-shot 微调、主表完全一致，否则三方对比不成立）：
    - 数据：MIT train（batch1，41 支）；测试集 batch2 一支不碰
    - 起点 0.5；上下文 512；chunk 64；真值 savgol(11,2) 平滑
    - 目标：跨 cell 的 MAE_smooth **中位数**（主表口径，抗离群 cell）

产物：
    03_实验结果/calib/calibration_head.json   完整网格 + 最优参数（MODELS 注册表读它）
    03_实验结果/calib/grid_surface.csv        网格曲面（s × b → 中位 MAE）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from adapters import get_forecaster, rolling_forecast_batch          # noqa: E402
from adapters.calibrated import CalibratedForecaster                 # noqa: E402
from configs.config import DATASETS, PROTOCOL, RESULTS_ROOT          # noqa: E402
from data.loaders import load_dataset                                # noqa: E402
from data.protocol import clean_series, relative_split               # noqa: E402
from evaluation.metrics import trajectory_metrics                    # noqa: E402

# 目标口径：主表用 MAE_smooth 的中位数（不是均值），离群 cell 不绑架结论
OBJ = "mae_smooth"


def parse_args():
    p = argparse.ArgumentParser(description="拟合 2 参数单调校准头")
    p.add_argument("--dataset", default="mit", choices=["mit", "nasa", "calce"],
                   help="拟合用数据集（默认 mit：与 few-shot 微调同一批数据）")
    p.add_argument("--split", default="train",
                   choices=["train", "test", "secondary_test", "all"],
                   help="默认 train；**不要**用 test，否则与零样本/微调的对比不干净")
    p.add_argument("--base", default="chronos-bolt",
                   help="被冻结的基础模型（chronos-bolt|chronos|timesfm|...）")
    p.add_argument("--chunk", type=int, default=PROTOCOL["rolling_chunk"])
    p.add_argument("--max-context", type=int, default=512,
                   help="与主表一致：统一夹到 512，避免用更长上下文换取不公平的优势")
    p.add_argument("--start-frac", type=float, default=PROTOCOL["start_frac"])
    p.add_argument("--fast", action="store_true",
                   help="小网格冒烟（3×3 粗 + 3×3 细），几分钟出结果")
    p.add_argument("--max-cells", type=int, default=0,
                   help="0=全部；冒烟时设小值（只影响拟合用多少 cell，不改变口径）")
    p.add_argument("--cells-from", default=None,
                   help="从 fewshot 的 *_train.json 读 support_cells，只在同一批电池上拟合。"
                        "用于和微调做**严格同数据**对照（微调因 CTX+HOR 长度限制只用了 19 支，"
                        "而本脚本默认能用满 41 支，不比一下会被质疑不公平）")
    p.add_argument("--out", default=None)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def build_tasks(dataset: str, split: str, frac: float):
    """与 run_benchmark.build_tasks 同一套清洗/切分/平滑口径。"""
    cells, meta = load_dataset(dataset, DATASETS, split=split, with_meta=True)
    tasks = []
    skipped = 0
    for cid, df in cells.items():
        df = clean_series(df, PROTOCOL["outlier_thresh"])
        if len(df) < 20:
            skipped += 1
            continue
        cell_eol = (meta.get(cid) or {}).get("cycle_life")
        sp = relative_split(
            df, frac=frac,
            smooth_window=PROTOCOL["smooth_window"],
            smooth_order=PROTOCOL["smooth_order"],
            ref_points=PROTOCOL["soh_ref_points"],
            eol_true=cell_eol,
        )
        if sp is None:
            continue
        tasks.append({"cell_id": cid, "context": sp.context,
                      "truth": sp.truth, "truth_smooth": sp.truth_smooth})
    if skipped:
        print(f"      跳过 {skipped} 个序列（清洗后 < 20 循环）")
    return tasks


def score(base, tasks: List[dict], chunk: int, s: float, b: float,
          want_per_cell: bool = False):
    """在给定 (s, b) 下跑一遍递归，返回跨 cell 的 MAE_smooth 统计。"""
    model = CalibratedForecaster(base, s=s, b=b)
    ctxs = [t["context"] for t in tasks]
    hors = [len(t["truth"]) for t in tasks]
    preds = rolling_forecast_batch(model, ctxs, hors, chunk)
    maes, per = [], {}
    for t, p in zip(tasks, preds):
        m = trajectory_metrics(p, t["truth"], t["truth_smooth"])
        maes.append(float(m[OBJ]))
        per[t["cell_id"]] = float(m[OBJ])
    maes = np.asarray(maes, dtype=float)
    out = {"s": s, "b": b, "n_cells": int(maes.size),
           "median_mae_smooth": float(np.median(maes)),
           "mean_mae_smooth": float(np.mean(maes)),
           "q25_mae_smooth": float(np.percentile(maes, 25)),
           "q75_mae_smooth": float(np.percentile(maes, 75))}
    if want_per_cell:
        out["per_cell"] = per
    return out


def grid_search(base, tasks, chunk, fast: bool, quiet: bool):
    """粗网格 → 在最优邻域细网格。两阶段各 ~1 个数量级的参数量，总耗时约 15 分钟。"""
    if fast:
        s_coarse = [0.9, 1.0, 1.1]
        b_coarse = [-0.5, 0.0, 0.5]
    else:
        # 覆盖面要够宽：SOH 每步衰减量级 ~0.01–0.1 SOH%/循环，b 到 ±3 已远超
        s_coarse = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        b_coarse = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

    def run(gs, bs, tag):
        recs = []
        for i, s in enumerate(gs):
            for b in bs:
                t0 = time.time()
                r = score(base, tasks, chunk, s, b)
                r["stage"] = tag
                recs.append(r)
                if not quiet:
                    print(f"      [{tag}] s={s:<5g} b={b:<6g} -> "
                          f"中位 MAE {r['median_mae_smooth']:.4f} "
                          f"({time.time() - t0:.1f}s)", flush=True)
        return recs

    recs = run(s_coarse, b_coarse, "coarse")
    best = min(recs, key=lambda r: r["median_mae_smooth"])

    # 细网格：围绕粗最优，步长取粗网格间距的 1/4
    ds = (s_coarse[1] - s_coarse[0]) / 4 if len(s_coarse) > 1 else 0.1
    db = (b_coarse[1] - b_coarse[0]) / 4 if len(b_coarse) > 1 else 0.5
    s_fine = [round(best["s"] + k * ds, 6) for k in (-2, -1, 0, 1, 2)]
    b_fine = [round(best["b"] + k * db, 6) for k in (-2, -1, 0, 1, 2)]
    s_fine = [x for x in s_fine if x > 0]
    if not quiet:
        print(f"      细网格：s∈{s_fine} b∈{b_fine}", flush=True)
    recs += run(s_fine, b_fine, "fine")

    best = min(recs, key=lambda r: r["median_mae_smooth"])
    return recs, best


def main():
    args = parse_args()
    t_start = time.time()
    out_dir = Path(args.out) if args.out else RESULTS_ROOT / "calib"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f">>> 拟合校准头：base={args.base} 数据={args.dataset}/{args.split} "
          f"frac={args.start_frac} ctx<={args.max_context} chunk={args.chunk}")
    tasks = build_tasks(args.dataset, args.split, args.start_frac)
    support = None
    if args.cells_from:
        support = json.loads(Path(args.cells_from).read_text(encoding="utf-8"))["support_cells"]
        keep = set(support)
        n_before = len(tasks)
        tasks = [t for t in tasks if t["cell_id"] in keep]
        print(f"      限定到 fewshot support_cells：{n_before} -> {len(tasks)} 支")
        missing = keep - {t["cell_id"] for t in tasks}
        if missing:
            print(f"      ⚠️ 有 {len(missing)} 支在切分后不可用：{sorted(missing)[:5]}")
    if args.max_cells:
        tasks = tasks[:args.max_cells]
    print(f"      {len(tasks)} 个拟合任务")

    base = get_forecaster(args.base)
    if args.max_context:
        base.max_context = int(args.max_context)
    if not base.supports_batch:
        print("      ⚠️ 基础模型不支持批量，拟合会明显变慢")

    recs, best = grid_search(base, tasks, args.chunk, args.fast, args.quiet)

    # 参照点：恒等校准 (s=1, b=0) 即零样本；它一定在粗网格里，这里显式取
    ident = min(recs, key=lambda r: (abs(r["s"] - 1.0), abs(r["b"])))
    base_ref = score(base, tasks, args.chunk, 1.0, 0.0, want_per_cell=True)
    best_ref = score(base, tasks, args.chunk, best["s"], best["b"], want_per_cell=True)

    delta = base_ref["median_mae_smooth"] - best_ref["median_mae_smooth"]
    rel = delta / base_ref["median_mae_smooth"] * 100 if base_ref["median_mae_smooth"] else 0.0

    print(f"\n=== 结果 ===")
    print(f"  零样本（s=1,b=0）   中位 MAE_smooth {base_ref['median_mae_smooth']:.4f} SOH%")
    print(f"  校准（s={best['s']:g}, b={best['b']:g}）  中位 MAE_smooth "
          f"{best_ref['median_mae_smooth']:.4f} SOH%")
    print(f"  绝对改善 {delta:+.4f} SOH%（{rel:+.2f}% 相对）")
    if delta <= 0:
        print("  ⚠️ 校准没有带来改善——机制假设（'敢不敢外推'）在该数据集上不成立，"
              "如实记录，不要挑网格。")

    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "objective": f"跨 cell 的 {OBJ} 中位数（主表口径）",
        "protocol": {"dataset": args.dataset, "split": args.split,
                     "start_frac": args.start_frac, "chunk": args.chunk,
                     "max_context": args.max_context,
                     "smooth_window": PROTOCOL["smooth_window"],
                     "smooth_order": PROTOCOL["smooth_order"],
                     "soh_ref_points": PROTOCOL["soh_ref_points"],
                     "outlier_thresh": PROTOCOL["outlier_thresh"],
                     "eol_threshold": PROTOCOL["eol_threshold"]},
        "base_model": args.base,
        "n_fit_cells": len(tasks),
        "fit_cells_from": args.cells_from,
        "fit_cell_ids": sorted(t["cell_id"] for t in tasks),
        "formula": "p' = c + s*(p - c) + b   (c = 上下文最后一个点，逐 chunk 施加)",
        "grid": {"coarse_s": sorted({r["s"] for r in recs if r["stage"] == "coarse"}),
                 "coarse_b": sorted({r["b"] for r in recs if r["stage"] == "coarse"}),
                 "stages": sorted({r["stage"] for r in recs})},
        "search": recs,
        "best": {"s": best["s"], "b": best["b"],
                 "median_mae_smooth": best_ref["median_mae_smooth"],
                 "mean_mae_smooth": best_ref["mean_mae_smooth"],
                 "per_cell_mae_smooth": best_ref["per_cell"]},
        "baseline_zero_shot": {"s": 1.0, "b": 0.0,
                               "median_mae_smooth": base_ref["median_mae_smooth"],
                               "mean_mae_smooth": base_ref["mean_mae_smooth"],
                               "per_cell_mae_smooth": base_ref["per_cell"]},
        "improvement": {"abs_mae_smooth": delta, "rel_pct": rel},
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    jpath = out_dir / "calibration_head.json"
    jpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 网格曲面（去掉 per_cell，太长）
    import csv
    cpath = out_dir / "grid_surface.csv"
    with cpath.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "s", "b", "n_cells", "median_mae_smooth",
                    "mean_mae_smooth", "q25_mae_smooth", "q75_mae_smooth"])
        for r in sorted(recs, key=lambda r: (r["stage"], r["s"], r["b"])):
            w.writerow([r["stage"], r["s"], r["b"], r["n_cells"],
                        round(r["median_mae_smooth"], 6), round(r["mean_mae_smooth"], 6),
                        round(r["q25_mae_smooth"], 6), round(r["q75_mae_smooth"], 6)])

    print(f"\n      参数已写入 {jpath}")
    print(f"      曲面已写入 {cpath}")
    print(f"      总耗时 {payload['elapsed_sec']}s")


if __name__ == "__main__":
    main()
