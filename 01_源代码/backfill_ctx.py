"""给历史 predictions_*.npz 回填上下文段（`key|ctx`），免重跑推理。

背景：run_benchmark.py 早期版本只存 (pred, truth)，figures.py 画轨迹图时
拿不到真实上下文，只能退化成一条水平直线（视觉上像"数据是假的"）。
本脚本按 summary.json 里记录的 dataset/split/start_fracs 重建同一批任务，
把 context 补进 npz。

用法：
    python backfill_ctx.py                 # 扫 03_实验结果 全部目录
    python backfill_ctx.py --run mit-zero-all-baselines
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from configs.config import DATASETS, RESULTS_ROOT          # noqa: E402
from data.loaders import load_dataset                       # noqa: E402
from run_benchmark import build_tasks                       # noqa: E402


def rebuild_contexts(dataset: str, split: str | None, fracs) -> dict:
    """返回 {f'{cell}@{frac}': context}"""
    cells, meta = load_dataset(dataset, DATASETS, split=split, with_meta=True)
    tasks = build_tasks(cells, meta, fracs, quiet=True)
    return {f"{t.cell_id}@{t.frac}": np.asarray(t.context, float) for t in tasks}


def backfill_dir(run_dir: Path, cache: dict, force: bool = False) -> int:
    sj = run_dir / "summary.json"
    npzs = sorted(run_dir.glob("predictions_*.npz"))
    if not sj.exists() or not npzs:
        return 0
    meta = json.loads(sj.read_text(encoding="utf-8"))
    meta = meta.get("meta") or meta          # 兼容两种布局
    dataset = meta.get("dataset")
    if not dataset:
        print(f"  !! {run_dir.name}: summary 缺 dataset，跳过")
        return 0
    split = meta.get("split")
    fracs = meta.get("start_fracs") or [0.5]

    ck = (dataset, split, tuple(fracs))
    if ck not in cache:
        print(f"  重建上下文：dataset={dataset} split={split} fracs={fracs} ...", flush=True)
        cache[ck] = rebuild_contexts(dataset, split, list(fracs))
    ctx_map = cache[ck]

    total = 0
    for f in npzs:
        with np.load(f) as z:
            data = {k: z[k] for k in z.files}
        missing = [k for k in data if "@" in k and "|ctx" not in k]
        if not missing:
            continue
        added = 0
        for k in missing:
            c = ctx_map.get(k)
            if c is None:
                continue
            data[f"{k}|ctx"] = c
            added += 1
        if added and (force or added):
            np.savez_compressed(f, **data)
            print(f"    {f.name}: 补 {added} 条上下文")
            total += added
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="只处理某个结果目录名")
    ap.add_argument("--force", action="store_true", help="已存在 |ctx 也覆盖重写")
    args = ap.parse_args()

    dirs = [RESULTS_ROOT / args.run] if args.run else sorted(
        p.parent for p in RESULTS_ROOT.glob("*/summary.json"))
    cache: dict = {}
    grand = 0
    for d in dirs:
        if not d.exists():
            raise SystemExit(f"目录不存在: {d}")
        print(f"[{d.name}]")
        grand += backfill_dir(d, cache, args.force)
    print(f"完成：共回填 {grand} 条上下文")


if __name__ == "__main__":
    main()
