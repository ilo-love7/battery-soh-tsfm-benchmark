"""analyze_eol2.py —— EOL 指标重设计分析（执行记录 §9.2）。

为什么要有这个脚本
------------------
v1 的 EOL 命中率在 MIT 上**恒为 0%**：数据在 SOH 跌到 80% 附近就截断，
"预测序列首次跌破 80%" 对零样本模型几乎不可能发生，于是这个指标在 MIT 上
不含任何信息量。但"没跌破"和"差 16 个 SOH 点没跌破"显然是两回事，
v1 把它们混成一个 0。

NASA 闸门后只剩 3 支、CALCE 8 支，也不够做显著性检验。

这个脚本不重跑任何推理，只读 `03_实验结果/prob/*/per_cell_*.csv`
（那里已经带了重设计后的字段），输出三张对照：

  1. `rul_rel_err`      RUL 相对误差 = |eol_pred − eol_true| / (eol_true − start_cycle)
                        相对于**起点剩余寿命**归一，跨 cell、跨库可比；
  2. `eol_overshoot_cyc` 带符号过冲 = eol_pred − eol_true。
                        正 = 预测偏晚（穿越点循环号更大，衰减被低估 = 保守），
                        负 = 预测偏早（衰减被高估 = 乐观/激进）；
  3. `no_cross_margin`  未穿越时预测末段 SOH 与 80% 的距离。
                        正是它让 MIT 的"0% 命中率"变成"平均还差 X 个 SOH 点"。

用法
----
    python analyze_eol2.py                # 全部
    python analyze_eol2.py --dataset mit  # 单库

产物
----
    03_实验结果/汇总_EOL重设计.csv
    04_文档报告/EOL重设计.md
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "03_实验结果"
OUT_CSV = RESULTS / "汇总_EOL重设计.csv"
OUT_MD = ROOT.parent / "04_文档报告" / "EOL重设计.md"

# 概率口径的结果目录（二级目录，report.py 扫不到，不污染主表）
PROB_DIRS = sorted(p for p in RESULTS.glob("prob/*") if p.is_dir())

ORDER = ["chronos-bolt", "chronos", "timesfm", "moirai", "lag-llama",
         "chronos-bolt-cal", "naive_linear", "naive_last", "exp_decay", "arima"]


def collect(dataset: str | None = None) -> pd.DataFrame:
    rows: List[Dict] = []
    for d in PROB_DIRS:
        for f in sorted(d.glob("per_cell_*.csv")):
            model = f.stem[len("per_cell_"):]
            try:
                df = pd.read_csv(f, encoding="utf-8-sig")
            except Exception:
                continue
            if df.empty or "cell_id" not in df.columns:
                continue
            ds = d.name.split("-")[0]
            if dataset and ds != dataset:
                continue
            n = len(df)

            def col(name):
                return pd.to_numeric(df[name], errors="coerce") if name in df.columns \
                    else pd.Series(np.nan, index=df.index)

            hit = col("eol_hit")
            over = col("eol_overshoot_cyc")
            rre = col("rul_rel_err")
            margin = col("no_cross_margin")
            defined = col("eol_defined")

            n_cross = int((hit == 1).sum())
            n_nocross = int((hit == 0).sum())
            rows.append({
                "数据集": ds,
                "模型": model,
                "n": n,
                "EOL可定义n": int((defined == 1).sum()),
                "命中n": n_cross,
                "未穿越n": n_nocross,
                # v1 口径，留着做对照
                "v1命中率": float(hit.mean()) if hit.notna().any() else np.nan,
                # 重设计口径
                "RUL相对误差中位": float(rre.median()) if rre.notna().any() else np.nan,
                "RUL相对误差n": int(rre.notna().sum()),
                "过冲中位(循环)": float(over.median()) if over.notna().any() else np.nan,
                "过冲均值(循环)": float(over.mean()) if over.notna().any() else np.nan,
                "未穿越裕度中位(SOH%)": float(margin.median()) if margin.notna().any() else np.nan,
                "未穿越裕度均值(SOH%)": float(margin.mean()) if margin.notna().any() else np.nan,
                "CRPS中位": float(pd.to_numeric(df["crps"], errors="coerce").median())
                if "crps" in df.columns else np.nan,
                "PICP中位": float(pd.to_numeric(df["picp"], errors="coerce").median())
                if "picp" in df.columns else np.nan,
                "MPIW中位": float(pd.to_numeric(df["mpiw"], errors="coerce").median())
                if "mpiw" in df.columns else np.nan,
            })
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    L = ["# EOL 指标重设计（自动生成）", "",
         "> 由 `01_源代码/analyze_eol2.py` 从 `03_实验结果/prob/*/per_cell_*.csv` 汇总，勿手改。", "",
         "v1 的 EOL 命中率在 MIT 上恒为 0%（数据在 EOL 处截断，零样本模型几乎不可能",
         "在观测窗内跌破 80%），这个指标因此不含信息量。重设计补三个量：", "",
         "- **RUL 相对误差** = |eol_pred − eol_true| / (eol_true − start_cycle)，相对起点剩余寿命归一；",
         "- **过冲循环数**（带符号，= eol_pred − eol_true）：正 = 预测偏晚"
         "（穿越点循环号更大，衰减被低估 = 保守），负 = 预测偏早"
         "（衰减被高估 = 乐观/激进）；",
         "- **未穿越裕度**：预测末段 SOH 与 80% 的距离，让 MIT 的 0% 变成可读的差距。", ""]
    for ds, g in df.groupby("数据集", sort=False):
        g = g.copy()
        g["_o"] = g["模型"].map(lambda m: ORDER.index(m) if m in ORDER else 99)
        g = g.sort_values(["_o"])
        L += [f"## {ds}", "",
              "| 模型 | n | EOL可定义 | 命中n | v1命中率 | RUL相对误差(中位) | "
              "过冲中位(循环) | 未穿越裕度中位(SOH%) |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            rre = "—" if pd.isna(r["RUL相对误差中位"]) else f"{r['RUL相对误差中位']:.3f}"
            over = "—" if pd.isna(r["过冲中位(循环)"]) else f"{r['过冲中位(循环)']:+.1f}"
            mg = ("—" if pd.isna(r["未穿越裕度中位(SOH%)"])
                  else f"{r['未穿越裕度中位(SOH%)']:.2f}")
            L.append(f"| {r['模型']} | {r['n']} | {r['EOL可定义n']} | {r['命中n']} | "
                     f"{r['v1命中率']:.0%} | {rre} | {over} | {mg} |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, choices=["mit", "nasa", "calce"])
    args = ap.parse_args()

    if not PROB_DIRS:
        raise SystemExit("没有找到 03_实验结果/prob/*，先跑 bash scripts/run_prob.sh")
    df = collect(args.dataset)
    if df.empty:
        raise SystemExit("prob/ 下没有可用的 per_cell CSV")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_MD.write_text(to_markdown(df), encoding="utf-8")
    print(f"{len(df)} 行 -> {OUT_CSV}")
    print(f"markdown -> {OUT_MD}")
    show = ["数据集", "模型", "n", "命中n", "v1命中率",
            "RUL相对误差中位", "过冲中位(循环)", "未穿越裕度中位(SOH%)"]
    print(df[show].to_string(index=False))


if __name__ == "__main__":
    main()
