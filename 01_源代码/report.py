"""把 03_实验结果 下所有 summary.json 汇总成一张主表（Markdown + CSV）。

用法：
    python report.py                 # 全部结果
    python report.py --dataset mit   # 只看某个库

口径说明（与方案 §4.4 一致）：
- MAE/RMSE 用**平滑真值**（Savitzky-Golay 11,2）的中位数，单位 SOH%
- EOL 误差 = |预测 EOL − 真值 EOL|，中位数，单位循环；真值 EOL 不可用的 cell 不计入
- EOL 命中率 = 预测序列在观测窗内触及 80% 的比例（基线普遍 0%）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT.parent / "03_实验结果"
OUT_MD = ROOT.parent / "04_文档报告" / "结果表.md"
OUT_CSV = RESULTS_ROOT / "汇总表.csv"

# 表格里的显示顺序（基础模型在前，基线在后）
ORDER = ["chronos-bolt", "chronos", "timesfm", "moirai", "lag-llama",
         "naive_linear", "naive_last", "exp_decay", "arima"]


KNOWN_TAGS = ("ctx512", "sens", "arimasmoke", "refactor")
# 这些前缀的目录是消融/子实验，产物放二级目录（report.py 只扫一级）本来扫不到；
# 列在这里是双保险——万一有人把它们提到一级目录，也不会混进主表。
_SKIP_PREFIXES = ("abl-", "ablations-")


def run_ts(meta: dict) -> str:
    """summary.json 里记录的时间戳，用于「同一键只保留最新一条」的比较。

    刻意**不用文件 mtime**：git 不保留 mtime，干净克隆后每个文件的 mtime 都约等于
    检出时刻，「保留最新」会退化成「取决于检出顺序」——本仓库确实因此翻过车：
    NASA 的 chronos 行被后跑的 `_det_b`（确定性复查副本）顶掉，naive_linear 行被
    gating 修复前的旧副本 `nasa-zero-naive_linear` 顶掉，导致别人 clone 下来重跑
    汇总得到的主表和仓库里存的不一致（EOL 列出现 14.3/20.6 两个版本）。

    `meta["timestamp"]` 是 `YYYY-MM-DDTHH:MM:SS` 的 ISO 字符串，字典序即时间序，
    且随文件内容一起被 git 提交，因此跨机器、跨 clone 稳定。
    """
    ts = meta.get("timestamp") or ""
    return ts if isinstance(ts, str) else ""


def run_tag(dir_name: str) -> str:
    """从结果目录名推断实验标签：mit-zero-chronos-bolt-sens -> 'sens'。"""
    for t in KNOWN_TAGS:
        if dir_name.endswith("-" + t):
            return t
    return "main"


_WARNED_KEYS: set = set()


def _warn_bad_key(dir_name: str, key: str) -> None:
    """summary.json 里出现无法解析成 (模型, 起点) 的键时告警一次。

    这类键通常是消融/子实验留下的（如 `naive_linear_half@half`），
    放进主表只会产生一行起点为 nan 的垃圾行。只警告不阻止，
    让作者自己决定该不该进主表。
    """
    if (dir_name, key) in _WARNED_KEYS:
        return
    _WARNED_KEYS.add((dir_name, key))
    print(f"  [report] 跳过无法解析的 summary 键：{dir_name} 中的 {key!r}")


def collect(dataset_filter: str | None = None, skip_tags: tuple = ()) -> List[Dict]:
    """扫所有 summary.json。同一 (数据集,划分,实验,模型,起点) 只保留**最新**一条，
    避免历史目录把表撑出重复行；不同实验标签（main/sens/ctx512）分开保留。

    「最新」按 meta["timestamp"] 判（见 run_ts）；时间戳相同或缺失时保留
    sorted(glob) 的先到者，因此输出只取决于目录名，与文件系统顺序无关。
    """
    best: Dict[tuple, tuple] = {}
    for f in sorted(RESULTS_ROOT.glob("*/summary.json")):
        if any(t in f.parent.name for t in skip_tags):
            continue
        if any(f.parent.name.startswith(p) for p in _SKIP_PREFIXES):
            print(f"  [report] 跳过消融/子实验目录：{f.parent.name}")
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = d.get("meta", {})
        ds = meta.get("dataset", "?")
        if dataset_filter and ds != dataset_filter:
            continue
        split = meta.get("split") or "all"
        exp = run_tag(f.parent.name)
        info = meta.get("model_info", {})
        for key, summ in (d.get("summary") or {}).items():
            model, _, frac = key.rpartition("@")
            if not model:                      # 没有 "@" 的畸形键
                _warn_bad_key(f.parent.name, key)
                continue
            try:
                frac_f = float(frac)
            except ValueError:
                # 例如消融实验写的 "naive_linear_half@half"：分子是标签不是起点，
                # 进主表只会得到一行 nan。跳过并告警，不静默吞掉。
                _warn_bad_key(f.parent.name, key)
                continue
            mi = info.get(model, {})
            eol_err = summ.get("eol_abs_err", {})
            eol_hit = summ.get("eol_hit", {})
            mae_s = summ.get("mae_smooth", {})
            row = {
                "数据集": ds,
                "划分": split,
                "实验": exp,
                "模型": model,
                "起点": frac_f,
                "n": mae_s.get("n", mi.get("n_ok", "")),      # 每格 n（不是整轮 n）
                "MAE_smooth": mae_s.get("median", float("nan")),
                "RMSE_smooth": summ.get("rmse_smooth", {}).get("median", float("nan")),
                "MAE_raw": summ.get("mae_raw", {}).get("median", float("nan")),
                "EOL命中率": eol_hit.get("mean", float("nan")),
                "EOL误差": eol_err.get("median", float("nan")),
                "EOL_n": eol_err.get("n", 0),
                "耗时s": mi.get("elapsed_sec", float("nan")),
                "目录": f.parent.name,
            }
            k = (ds, split, exp, model, row["起点"])
            ts = run_ts(meta)
            # 严格大于才替换：时间戳相同（或都缺省）时保留 sorted(glob) 的先到者，
            # 保证输出只由目录名决定，与文件 mtime / clone 检出顺序无关。
            if k not in best or ts > best[k][0]:
                best[k] = (ts, row)
    return [v[1] for v in best.values()]


def to_markdown(df: pd.DataFrame) -> str:
    lines = ["# 基准结果主表（自动生成）", "",
             "> 由 `01_源代码/report.py` 从 `03_实验结果/*/summary.json` 汇总，勿手改。",
             "> MAE/RMSE 为**平滑真值**下的中位数（SOH%），越小越好；"
             "EOL 误差单位=循环，EOL 命中率=预测在观测窗内触及 80% 的比例。", ""]
    for (ds, split, exp), g in df.groupby(["数据集", "划分", "实验"], sort=False):
        g = g.copy()
        g["_o"] = g["模型"].map(lambda m: ORDER.index(m) if m in ORDER else 99)
        g = g.sort_values(["起点", "_o"])
        title = f"## {ds} · {split}"
        if exp != "main":
            title += f" · {exp}"
        lines.append(title)
        lines.append("")
        lines.append("| 模型 | 起点 | n | MAE_smooth | RMSE_smooth | EOL 命中率 | EOL 误差(循环) | EOL_n | 耗时(s) |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in g.iterrows():
            eol_err = "—" if pd.isna(r["EOL误差"]) else f"{r['EOL误差']:.1f}"
            hit = "—" if pd.isna(r["EOL命中率"]) else f"{r['EOL命中率']:.0%}"
            # 耗时缺失（早期合并批次没记 runtime）时显示 —，不要印出 nan
            cost = "—" if pd.isna(r["耗时s"]) else f"{r['耗时s']:.1f}"
            lines.append(
                f"| {r['模型']} | {r['起点']:.2f} | {r['n']} | {r['MAE_smooth']:.3f} | "
                f"{r['RMSE_smooth']:.3f} | {hit} | {eol_err} | {r['EOL_n']} | "
                f"{cost} |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--skip", nargs="*",
                    default=["arimasmoke", "refactor", "_det_"],
                    help="跳过目录名含这些子串的结果"
                         "（冒烟 / 重构验证 / 确定性复查副本，后者的指标与正式轮完全"
                         "相同，留着只会把主表里的口径搅乱）")
    args = ap.parse_args()

    rows = collect(args.dataset, skip_tags=tuple(args.skip))
    if not rows:
        raise SystemExit("没有找到任何 summary.json，先跑 run_benchmark.py")
    df = pd.DataFrame(rows).sort_values(["数据集", "划分", "实验", "起点", "模型"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_MD.write_text(to_markdown(df), encoding="utf-8")
    print(f"{len(df)} 行 -> {OUT_CSV}")
    print(f"markdown -> {OUT_MD}")
    print(df[["数据集", "模型", "起点", "n", "MAE_smooth", "EOL命中率"]].to_string(index=False))


if __name__ == "__main__":
    main()
