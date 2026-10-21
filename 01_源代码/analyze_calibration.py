"""analyze_calibration.py —— 校准头三方对比（执行记录 §9.3）。

不重跑任何推理，只汇总已有产物：

    A. 零样本     03_实验结果/prob/<ds>-*/per_cell_<model>.csv
    B. 校准头     03_实验结果/prob/<ds>-chronos-bolt-cal*/per_cell_chronos-bolt-cal.csv
                  + 03_实验结果/calib/calibration_head.json（参数与拟合代价）
    C. 全量微调   03_实验结果/fewshot_eval/mit-zero-chronos-bolt-ft*/summary.json
                  （k=1/4/8/19，见 fewshot_finetune.py）

输出两张表：
    1. MIT test 上的四档对比（零样本 / 2 参数 / 微调 k=1,4,8,19）
    2. 跨库迁移（MIT 上拟合的 (s,b) 直接用到 NASA / CALCE，不做任何重拟合）

⚠️ 为什么要有 `cal19` 这一档：微调因 `len(df) < ctx+horizon` 的长度门槛只用到
19 支电池，而校准头默认能用满 41 支。不比一下会被质疑"数据预算不公平"。
cal19 就是只在同一 19 支上拟合的校准头，用来做严格同数据对照。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "03_实验结果"
PROB = RESULTS / "prob"
FEW = RESULTS / "fewshot_eval"
OUT_CSV = RESULTS / "汇总_校准头.csv"
OUT_MD = ROOT.parent / "04_文档报告" / "校准头对比.md"

DATASETS = [("mit", "MIT batch2（43 支）"), ("nasa", "NASA（4 支）"),
            ("calce", "CALCE（14 支）")]


def read_per_cell(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty or "mae_smooth" not in df.columns:
        return None
    return df


def find_cal_dir(ds: str) -> Path | None:
    """校准目录名带或不带 -ctx512 后缀（MIT 走 ctx512 公平对照）。"""
    for name in (f"{ds}-chronos-bolt-cal-ctx512", f"{ds}-chronos-bolt-cal"):
        p = PROB / name
        if p.is_dir():
            return p
    return None


def summarize(df: pd.DataFrame) -> Dict[str, float]:
    out = {"n": len(df)}
    for c in ("mae_smooth", "rmse_smooth", "crps", "picp", "mpiw", "pinaw",
              "eol_overshoot_cyc", "rul_rel_err", "no_cross_margin"):
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                out[c] = float(s.median())
    # ⚠️ eol_hit 是 0/1 列，必须取**均值**（= 命中率）。
    # 取中位数会得到 0（只要命中数不到一半），曾把校准模型的 47% 显示成 0%。
    if "eol_hit" in df.columns:
        s = pd.to_numeric(df["eol_hit"], errors="coerce")
        if s.notna().any():
            out["eol_hit"] = float(s.mean())
    return out


def cross_dataset() -> pd.DataFrame:
    rows = []
    for ds, label in DATASETS:
        # 零样本：优先 ctx512 版本（MIT），否则原生
        z = None
        for name in (f"{ds}-chronos-bolt-chronos-timesfm",):
            p = PROB / name
            if p.is_dir():
                z = read_per_cell(p / "per_cell_chronos-bolt.csv")
        cal_dir = find_cal_dir(ds)
        c = read_per_cell(cal_dir / "per_cell_chronos-bolt-cal.csv") if cal_dir else None
        if z is None or c is None:
            continue
        sz, sc = summarize(z), summarize(c)
        rows.append({
            "数据集": ds, "说明": label, "n": sz["n"],
            "零样本_MAE": sz.get("mae_smooth", np.nan),
            "校准_MAE": sc.get("mae_smooth", np.nan),
            "ΔMAE": sc.get("mae_smooth", np.nan) - sz.get("mae_smooth", np.nan),
            "ΔMAE%": (sc.get("mae_smooth", np.nan) / sz.get("mae_smooth", np.nan) - 1) * 100,
            "零样本_CRPS": sz.get("crps", np.nan), "校准_CRPS": sc.get("crps", np.nan),
            "零样本_PICP": sz.get("picp", np.nan), "校准_PICP": sc.get("picp", np.nan),
            "零样本_MPIW": sz.get("mpiw", np.nan), "校准_MPIW": sc.get("mpiw", np.nan),
            "零样本_EOL命中": sz.get("eol_hit", np.nan), "校准_EOL命中": sc.get("eol_hit", np.nan),
            "零样本_未穿越裕度": sz.get("no_cross_margin", np.nan),
            "校准_RUL相对误差": sc.get("rul_rel_err", np.nan),
        })
    return pd.DataFrame(rows)


def mit_ladder() -> pd.DataFrame:
    """MIT test 上：零样本 → 2 参数 → 微调 k=1/4/8/19。"""
    rows = []
    # 零样本（ctx512）
    z = None
    for name in ("mit-chronos-bolt-chronos-timesfm",):
        z = read_per_cell(PROB / name / "per_cell_chronos-bolt.csv")
    if z is not None:
        s = summarize(z)
        rows.append({"方案": "零样本 Chronos-Bolt", "类别": "零样本",
                     "预算": "0 参数 / 0 支", "MAE": s.get("mae_smooth"),
                     "EOL命中": s.get("eol_hit"), "n": s["n"]})
    # 2 参数（41 支）
    cal_dir = find_cal_dir("mit")
    c41 = read_per_cell(cal_dir / "per_cell_chronos-bolt-cal.csv") if cal_dir else None
    if c41 is not None:
        s = summarize(c41)
        rows.append({"方案": "2 参数校准头（41 支拟合）", "类别": "校准",
                     "预算": "2 标量 / 41 支 / 无梯度", "MAE": s.get("mae_smooth"),
                     "EOL命中": s.get("eol_hit"), "n": s["n"]})
    # 2 参数（同 19 支，与微调 ftall 同数据）
    c19 = read_per_cell(FEW / "mit-zero-chronos-bolt-cal19" / "per_cell_chronos-bolt-cal.csv")
    if c19 is not None:
        s = summarize(c19)
        rows.append({"方案": "2 参数校准头（19 支，同微调数据）", "类别": "校准",
                     "预算": "2 标量 / 19 支 / 无梯度", "MAE": s.get("mae_smooth"),
                     "EOL命中": s.get("eol_hit"), "n": s["n"]})
    # 微调各档
    for k, tag in ((1, "ft1"), (4, "ft4"), (8, "ft8"), (19, "ftall")):
        f = FEW / f"mit-zero-chronos-bolt-{tag}" / "summary.json"
        if not f.exists():
            continue
        summ = json.loads(f.read_text(encoding="utf-8"))["summary"]
        for key, v in summ.items():
            if "mae_smooth" not in v:
                continue
            rows.append({"方案": f"微调 {k} 支（{tag}）", "类别": "微调",
                         "预算": f"~60M 权重 / {k} 支 / 300 步",
                         "MAE": v["mae_smooth"].get("median"),
                         "EOL命中": v.get("eol_hit", {}).get("mean"),
                         "n": v.get("mae_smooth", {}).get("n")})
    df = pd.DataFrame(rows)
    base = df.loc[df["类别"] == "零样本", "MAE"]
    if not base.empty:
        df["相对零样本%"] = (df["MAE"] / float(base.iloc[0]) - 1) * 100
    return df


def cal_params_table() -> pd.DataFrame:
    rows = []
    for label, p in (("41 支（默认）", RESULTS / "calib" / "calibration_head.json"),
                     ("19 支（同微调）", RESULTS / "calib" / "k19" / "calibration_head.json")):
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append({"拟合集": label, "s": d["best"]["s"], "b": d["best"]["b"],
                     "拟合集内中位 MAE": d["best"]["median_mae_smooth"],
                     "零样本同集": d["baseline_zero_shot"]["median_mae_smooth"],
                     "改善%": -d["improvement"]["rel_pct"],
                     "n_cells": d["n_fit_cells"],
                     "网格评估次数": len(d["search"])})
    return pd.DataFrame(rows)


def to_markdown(cross: pd.DataFrame, ladder: pd.DataFrame,
                params: pd.DataFrame) -> str:
    L = ["# 校准头对比（自动生成）", "",
         "> 由 `01_源代码/analyze_calibration.py` 汇总，勿手改。",
         "> 公式 `p' = c + s·(p − c) + b`，`c` = 上下文最后一个点，逐 chunk 施加。", ""]
    if not params.empty:
        L += ["## 拟合出的参数", "",
              "| 拟合集 | s | b | 拟合集内中位 MAE | 同集零样本 | 改善 | 支数 | 网格评估 |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in params.iterrows():
            L.append(f"| {r['拟合集']} | {r['s']:g} | {r['b']:g} | "
                     f"{r['拟合集内中位 MAE']:.3f} | {r['零样本同集']:.3f} | "
                     f"{r['改善%']:+.1f}% | {int(r['n_cells'])} | {int(r['网格评估次数'])} |")
        L.append("")
    if not ladder.empty:
        L += ["## MIT test：零样本 → 2 参数 → 微调", "",
              "| 方案 | 类别 | 数据/参数预算 | MAE_smooth | 相对零样本 | EOL 命中率 |",
              "|---|---|---|---|---|---|"]
        for _, r in ladder.iterrows():
            rel = "—" if (pd.isna(r.get("相对零样本%")) or r["类别"] == "零样本") \
                else f"{r['相对零样本%']:+.1f}%"
            hit = "—" if pd.isna(r.get("EOL命中")) else f"{r['EOL命中']:.0%}"
            L.append(f"| {r['方案']} | {r['类别']} | {r['预算']} | {r['MAE']:.3f} | "
                     f"{rel} | {hit} |")
        L.append("")
        L.append("零样本 = 冻结的 Chronos-Bolt，不做任何调整；EOL 命中率 = 预测在观测窗内"
                 "首次跌破 80% 的 cell 占比（MIT 数据在 EOL 处截断，零样本几乎不可能触及）。")
        L.append("")
    if not cross.empty:
        L += ["## 跨库：MIT 上拟合的 (s, b) 直接外推（不重拟合）", "",
              "全部统一 ctx512 口径（与 `prob/` 概率矩阵一致，CRPS/PICP 才可比）。", "",
              "| 数据集 | n | 零样本 MAE | 校准 MAE | Δ% | 零样本 CRPS | 校准 CRPS | "
              "零样本 PICP | 校准 PICP |",
              "|---|---|---|---|---|---|---|---|---|"]
        for _, r in cross.iterrows():
            L.append(f"| {r['数据集']} | {int(r['n'])} | {r['零样本_MAE']:.3f} | "
                     f"{r['校准_MAE']:.3f} | {r['ΔMAE%']:+.1f}% | "
                     f"{r['零样本_CRPS']:.3f} | {r['校准_CRPS']:.3f} | "
                     f"{r['零样本_PICP']:.1%} | {r['校准_PICP']:.1%} |")
        L.append("")
        L.append("PICP 名义值 80%（分位数层 0.1/0.9）。零样本全部远低于名义值 ⇒ "
                 "基础模型对未见过的衰减形态系统性**过度自信**。")
        L.append("")
    return "\n".join(L)


def main():
    cross = cross_dataset()
    ladder = mit_ladder()
    params = cal_params_table()
    if cross.empty and ladder.empty:
        raise SystemExit("没有可汇总的产物：先跑 scripts/run_prob.sh 与 scripts/run_cal.sh")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([cross.assign(表="跨库"), ladder.assign(表="MIT阶梯"),
               params.assign(表="参数")], ignore_index=True).to_csv(
        OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_MD.write_text(to_markdown(cross, ladder, params), encoding="utf-8")
    print(f"-> {OUT_CSV}")
    print(f"-> {OUT_MD}\n")
    if not params.empty:
        print(params.to_string(index=False))
        print()
    if not ladder.empty:
        print(ladder.to_string(index=False))
        print()
    if not cross.empty:
        print(cross[["数据集", "n", "零样本_MAE", "校准_MAE", "ΔMAE%",
                     "零样本_CRPS", "校准_CRPS", "零样本_PICP", "校准_PICP"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
