"""naive_linear 拟合窗口消融：最近 ⌊n/2⌋ 点（本文口径） vs 整段上下文。

动机
----
README / 短文 §3.2 / 执行记录都引用了三个消融数字：
    MIT 4.066 → 4.528    （本文版本更好）
    CALCE 4.623 → 6.275  （本文版本更好）
    NASA 6.111 → 2.478   （整段反而好 2.5 倍）
但这三个数此前是临时算的，仓库里既没有脚本也没有产物，读者无法验证。
本脚本把它固化为可复现实验，并用它与已发表数字对账。

口径与 run_benchmark.py 完全一致（同一套 load → clean → split → predict → 指标），
只把 NaiveLinear 的拟合窗口从 `n // 2` 换成 `n`。

产物：
    03_实验结果/ablations/naive_linear-fit/summary.json
    03_实验结果/ablations/naive_linear-fit/per_cell_<dataset>_<variant>.csv
    03_实验结果/汇总_naive_linear_拟合窗口消融.csv

⚠️ 故意放在二级目录：report.py 只 glob `*/summary.json`（一级），
一级目录里的消融结果会被扫进主表、并把 `模型@起点` 键弄成
`naive_linear_half@half` 这种解析不了的形状。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from adapters.base import Forecaster  # noqa: E402
from configs.config import DATASETS, PROTOCOL, RESULTS_ROOT  # noqa: E402
from data.loaders import load_dataset  # noqa: E402
from data.protocol import clean_series, relative_split  # noqa: E402
from evaluation.metrics import lifetime_metrics, summarize, trajectory_metrics  # noqa: E402


class NaiveLinearK(Forecaster):
    """naive_linear 的参数化版本：用上下文最近 k 个点做最小二乘外推。

    k = n // 2 即本文口径；k = n 即"整段拟合"。
    """

    def __init__(self, k_mode: str):
        self.k_mode = k_mode          # "half" | "full"
        self.name = f"naive_linear_{k_mode}"

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        n = len(context)
        if n < 2:
            return np.full(horizon, float(context[-1]))
        k = n if self.k_mode == "full" else max(2, n // 2)
        x = np.arange(n, dtype=float)
        b, a = np.polyfit(x[-k:], context[-k:], 1)
        xf = np.arange(n, n + horizon, dtype=float)
        return a + b * xf


def build_tasks(dataset: str, split, frac: float):
    cells, meta = load_dataset(dataset, DATASETS, split=split, with_meta=True)
    tasks = []
    for cid, df in cells.items():
        df = clean_series(df, PROTOCOL["outlier_thresh"])
        if len(df) < 20:
            continue
        cell_eol = (meta.get(cid) or {}).get("cycle_life")
        sp = relative_split(df, frac=frac,
                            smooth_window=PROTOCOL["smooth_window"],
                            smooth_order=PROTOCOL["smooth_order"],
                            ref_points=PROTOCOL["soh_ref_points"],
                            eol_true=cell_eol)
        if sp is None:
            continue
        tasks.append((cid, sp))
    return tasks


def run_variant(model: Forecaster, tasks, chunk: int) -> pd.DataFrame:
    recs = []
    for cid, sp in tasks:
        ctx = np.asarray(sp.context, float)
        horizon = len(sp.truth)
        # 递归滚动（与 base.rolling_forecast 同逻辑，基线无批处理）
        remaining, out = horizon, []
        work = ctx.copy()
        while remaining > 0:
            step = min(chunk, remaining)
            p = np.asarray(model.predict(work, step), float)
            out.append(p)
            work = np.concatenate([work, p])
            remaining -= step
        pred = np.concatenate(out)
        rec = {"cell_id": cid, "start_frac": 0.5,
               "n_context": len(ctx), "n_horizon": horizon}
        rec.update(trajectory_metrics(pred, sp.truth, sp.truth_smooth))
        rec.update(lifetime_metrics(pred, sp.cycles, sp.truth, sp.start_cycle,
                                    PROTOCOL["eol_threshold"], eol_true=sp.eol_true))
        recs.append(rec)
    return pd.DataFrame(recs)


def main():
    # 二级目录：不进 report.py 的一级 glob，避免污染主表
    out_root = RESULTS_ROOT / "ablations" / "naive_linear-fit"
    out_root.mkdir(parents=True, exist_ok=True)

    # 已发表数字（README / 短文 §3.2 / 执行记录 §五之六）
    PUBLISHED = {"mit": 4.066, "calce": 4.623, "nasa": 6.111}
    PUBLISHED_FULL = {"mit": 4.528, "calce": 6.275, "nasa": 2.478}

    specs = [("mit", "test"), ("calce", None), ("nasa", None)]
    summary, rows = {}, []
    for ds, split in specs:
        tasks = build_tasks(ds, split, PROTOCOL["start_frac"])
        print(f"\n=== {ds}（{split or 'all'}，{len(tasks)} cells）===")
        for mode in ("half", "full"):
            df = run_variant(NaiveLinearK(mode), tasks, PROTOCOL["rolling_chunk"])
            df.to_csv(out_root / f"per_cell_{ds}_{mode}.csv",
                      index=False, encoding="utf-8-sig")
            summ = summarize(df.drop(columns=["cell_id", "start_frac"],
                                     errors="ignore").to_dict("records"))
            med = summ["mae_smooth"]["median"]
            summary[f"{ds}@{mode}"] = summ
            ref = PUBLISHED if mode == "half" else PUBLISHED_FULL
            ok = "✓ 与已发表一致" if abs(med - ref[ds]) < 5e-4 else f"✗ 与已发表 {ref[ds]} 不符"
            print(f"  {mode:5s} MAE_smooth 中位 = {med:.6f}   {ok}")
            rows.append({"dataset": ds, "split": split or "all", "variant": mode,
                         "n": summ["mae_smooth"]["n"], "mae_smooth": med,
                         "rmse_smooth": summ["rmse_smooth"]["median"],
                         "published": ref[ds],
                         "match": abs(med - ref[ds]) < 5e-4})

    meta = {"protocol": PROTOCOL, "chunk": PROTOCOL["rolling_chunk"],
            "variants": {"half": "上下文最近 n // 2 个点（本文口径）",
                         "full": "整个上下文最小二乘"},
            "published_reference": {"half": PUBLISHED, "full": PUBLISHED_FULL},
            "note": "消融固化：此前这三个整段拟合数字是临时算的，仓库无脚本无产物"}
    (out_root / "summary.json").write_text(
        json.dumps({"meta": meta, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    rdf = pd.DataFrame(rows)
    rdf.to_csv(RESULTS_ROOT / "汇总_naive_linear_拟合窗口消融.csv",
               index=False, encoding="utf-8-sig")
    print(f"\n已保存 {out_root / 'summary.json'}")
    print(f"已保存 {RESULTS_ROOT / '汇总_naive_linear_拟合窗口消融.csv'}")

    n_ok = int(rdf["match"].sum())
    print(f"\n对账：{n_ok}/{len(rdf)} 与已发表数字一致")
    if n_ok != len(rdf):
        print("!! 有不一致的项，必须查清后再对外引用")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
