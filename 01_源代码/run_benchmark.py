"""时序基础模型零样本电池容量衰减基准 —— 主入口（真实数据版）。

用法：
    # 单模型
    python run_benchmark.py --model chronos-bolt --dataset mit --split test
    python run_benchmark.py --model naive_linear --dataset mit

    # 一次跑多个模型（共享同一份切分，避免重复清洗）
    python run_benchmark.py --models chronos-bolt,chronos,naive_linear --dataset mit

    # 敏感性分析（起点 40/50/60%）
    python run_benchmark.py --model chronos-bolt --dataset mit --start-fracs 0.4 0.5 0.6

设计红线：
- 依赖缺失抛 NotInstalledError 并退出，**绝不静默降级为随机数**（原版项目就死在这条）；
- 数据来自真实文件（MIT=阶段0 parquet，NASA=.mat，CALCE=zip 内 xlsx），零复制；
- 指标在 SOH 空间计算，含 EOL/RUL 误差与 EOL 命中率，不只看 RMSE；
- 递归预测按协议 H=chunk 滚动，不做一次硬预测（Chronos 单次上限 64，硬预测本就不可能）。
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

# ⚠️ Windows 专有坑：pandas / pyarrow 只要**先于** torch 导入，torch 就会报
#     OSError: [WinError 1114] 动态链接库(DLL)初始化例程失败 (c10.dll)
# （实测 import pandas → import torch 必失败，import numpy → import torch 正常）。
# 所以必须在 import pandas **之前**抢先把 torch 拉起来。只对 torch 系模型生效，
# 跑纯基线时不付这个开销。设 BENCH_NO_TORCH_PREIMPORT=1 可关闭。
_TORCH_MODELS = ("chronos-bolt", "chronos", "timesfm", "moirai", "lag-llama")
if not __import__("os").environ.get("BENCH_NO_TORCH_PREIMPORT") and any(
        m in a for a in sys.argv for m in _TORCH_MODELS):
    try:
        import torch  # noqa: F401
    except ImportError:
        pass

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from adapters import (NotInstalledError, get_forecaster,      # noqa: E402
                      rolling_forecast, rolling_forecast_batch,
                      rolling_forecast_quantiles_batch)
from configs.config import (DATASETS, PROTOCOL, RESULTS_ROOT,  # noqa: E402
                            check_dataset)
from data.loaders import load_dataset                           # noqa: E402
from data.protocol import clean_series, relative_split          # noqa: E402
from evaluation.metrics import (DEFAULT_LEVELS, lifetime_metrics,  # noqa: E402
                                median_deviation, probabilistic_metrics,
                                summarize, trajectory_metrics)

# 写进 summary.json 的包版本（验收标准 #5：可追溯）
_TRACKED_PKGS = ["torch", "transformers", "chronos-forecasting", "timesfm",
                 "uni2ts", "gluonts", "numpy", "pandas", "scipy", "statsmodels"]


def parse_args():
    p = argparse.ArgumentParser(description="电池容量衰减零样本基准")
    p.add_argument("--model", default=None,
                   help="单模型：chronos-bolt|chronos|timesfm|moirai|lag-llama|naive_last|naive_linear|exp_decay|arima|all-baselines")
    p.add_argument("--models", default=None,
                   help="逗号分隔多模型，如 chronos-bolt,chronos,naive_linear")
    p.add_argument("--dataset", required=True, choices=["mit", "nasa", "calce"])
    p.add_argument("--setting", default="zero", choices=["zero", "few"],
                   help="zero=零样本；few=少样本微调（尚未实现，会显式报错）")
    p.add_argument("--start-fracs", type=float, nargs="+", default=[PROTOCOL["start_frac"]])
    p.add_argument("--split", default="test",
                   choices=["train", "test", "secondary_test", "all"],
                   help="仅 MIT 有效：阶段 0 的 batch1/2/3 三划分")
    p.add_argument("--max-cells", type=int, default=0, help="0=全部；调试时设小值")
    p.add_argument("--chunk", type=int, default=PROTOCOL["rolling_chunk"],
                   help="递归步长（会被模型 max_horizon 夹紧）")
    p.add_argument("--max-context", type=int, default=0,
                   help="统一把各模型上下文上限夹到该值（0=用模型自身上限）。"
                        "用于公平性对照：TimesFM 只有 512，MIT 长序列有 2235")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--tag", default="", help="结果目录后缀，用于区分消融实验")
    p.add_argument("--seed", type=int, default=None,
                   help="采样型基础模型的随机种子（默认用适配器内置值；多种子方差分析时显式指定）")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--quantiles", type=float, nargs="*", default=None,
                   metavar="LEVEL",
                   help="分位数层，如 --quantiles 0.1 0.5 0.9（默认模型支持时就开）。"
                        "传空串 --quantiles '' 可显式关闭。基线不支持分位数，自动跳过。")
    return p.parse_args()


def resolve_levels(args) -> List[float]:
    """--quantiles 的三种形态：不给=None（模型支持则用默认层）、
    给数字=用这些层、给空串=显式关闭。"""
    if args.quantiles is None:
        return list(DEFAULT_LEVELS)
    if len(args.quantiles) == 0:
        return []
    lv = sorted(float(x) for x in args.quantiles)
    if not 0.0 < lv[0] or lv[-1] >= 1.0:
        raise SystemExit(f"分位数层必须在 (0,1) 开区间内：{lv}")
    if len(set(lv)) != len(lv):
        raise SystemExit(f"分位数层有重复：{lv}")
    return lv


def resolve_model_names(args) -> List[str]:
    if args.models:
        names = [s.strip() for s in args.models.split(",") if s.strip()]
    elif args.model:
        names = [args.model]
    else:
        raise SystemExit("必须给 --model 或 --models")
    if "all-baselines" in names:
        names = [n for n in names if n != "all-baselines"] + \
                ["naive_last", "naive_linear", "exp_decay", "arima"]
    return names


def package_versions() -> Dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for pkg in _TRACKED_PKGS:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not-installed"
    return out


@dataclass
class Task:
    cell_id: str
    frac: float
    context: np.ndarray
    truth: np.ndarray
    truth_smooth: np.ndarray
    cycles: np.ndarray
    start_cycle: int
    eol_true: float | None


def build_tasks(cells: dict, meta: dict, fracs: List[float],
                quiet: bool = False) -> List[Task]:
    """清洗 + 切分一次，所有模型共享（避免重复 IO 与重复平滑）。"""
    tasks: List[Task] = []
    skipped_short = 0
    for cid, df in cells.items():
        df = clean_series(df, PROTOCOL["outlier_thresh"])
        if len(df) < 20:
            skipped_short += 1
            continue
        cell_eol = (meta.get(cid) or {}).get("cycle_life")
        for frac in fracs:
            sp = relative_split(
                df, frac=frac,
                smooth_window=PROTOCOL["smooth_window"],
                smooth_order=PROTOCOL["smooth_order"],
                ref_points=PROTOCOL["soh_ref_points"],
                eol_true=cell_eol,
            )
            if sp is None:
                continue
            tasks.append(Task(cid, frac, sp.context, sp.truth, sp.truth_smooth,
                              sp.cycles, sp.start_cycle, sp.eol_true))
    if skipped_short and not quiet:
        print(f"      跳过 {skipped_short} 个序列（清洗后 < 20 循环）")
    return tasks


def run_model(model, tasks: List[Task], chunk: int, quiet: bool,
              levels: Sequence[float] = DEFAULT_LEVELS):
    """批量推理 + 逐任务算指标。单个任务失败只记 error，不拖垮整轮。

    ⚠️ 按 `frac` **分组**推理，不把不同起点混进同一批。
    原因：Chronos-T5 是"先采样 num_samples 条再取分位"的模型，种子按批设置，
    同一个 (cell, frac) 的抽样结果会随"这批里还有谁"而变。
    实测 NASA 上 B0005@0.5 在 4 任务批里 MAE 0.80、在 12 任务批里 1.03，
    直接导致主表（单起点）与敏感性表（三起点）的 0.5 行对不上。
    分组后同一 frac 的批组成只取决于该 split 的 cell 集合，两次跑才可比。

    分位数：模型 supports_quantiles() 时**额外**跑一遍分位数递归，
    与点预测共用同一批任务、同一分组、同一个 chunk。两条链路互不影响：
    点预测的口径与 v1 逐位一致（否则已发表数字就废了）。
    """
    t0 = time.time()
    preds: List[np.ndarray | None] = [None] * len(tasks)
    errors: Dict[int, str] = {}
    # (Q, H) 分位数预测；模型不支持或没开 --quantiles 时保持 None
    qpreds: List[np.ndarray | None] = [None] * len(tasks)
    want_q = bool(levels) and model.supports_quantiles()

    groups: Dict[float, List[int]] = {}
    for i, t in enumerate(tasks):
        groups.setdefault(t.frac, []).append(i)

    for frac in sorted(groups):
        idxs = groups[frac]
        contexts = [tasks[i].context for i in idxs]
        horizons = [len(tasks[i].truth) for i in idxs]
        # 采样型模型的随机状态标记（确定性模型返回 None）。
        # 分位数链路必须从**与点预测链路相同**的状态出发，否则两遍遍历的种子错开，
        # 采到不同样本 ⇒ q50 ≠ 点预测 ⇒ 轨迹分叉（详见 adapters.base.rng_checkpoint）。
        rng_ckpt = model.rng_checkpoint()
        try:
            out = rolling_forecast_batch(model, contexts, horizons, chunk)
            for j, i in enumerate(idxs):
                preds[i] = out[j]
        except Exception as e:
            # 批量失败 → 退回逐条，定位到具体 cell，避免整轮报废
            if not quiet:
                print(f"      批量推理失败（frac={frac:.2f}，{type(e).__name__}: {e}），退回逐条")
            for i in idxs:
                try:
                    preds[i] = rolling_forecast(model, tasks[i].context,
                                                len(tasks[i].truth), chunk)
                except NotInstalledError:
                    raise
                except Exception as e2:
                    errors[i] = f"{type(e2).__name__}: {e2}"

        if not want_q:
            continue
        # 分位数链路：同样按 frac 分组；失败只影响概率指标，不影响点指标
        model.rng_restore(rng_ckpt)
        try:
            qout = rolling_forecast_quantiles_batch(model, contexts, horizons,
                                                    chunk, list(levels))
            for j, i in enumerate(idxs):
                if i not in errors:
                    qpreds[i] = qout[j]
        except Exception as e:
            if not quiet:
                print(f"      分位数推理失败（frac={frac:.2f}，{type(e).__name__}: {e}），"
                      f"该轮概率指标留空")

    elapsed = time.time() - t0
    records = []
    for i, t in enumerate(tasks):
        if i in errors:
            records.append({"cell_id": t.cell_id, "start_frac": t.frac,
                            "error": errors[i]})
            continue
        pred = preds[i]
        rec = {"cell_id": t.cell_id, "start_frac": t.frac,
               "n_context": len(t.context), "n_horizon": len(t.truth),
               "start_cycle": int(t.start_cycle)}
        rec.update(trajectory_metrics(pred, t.truth, t.truth_smooth))
        rec.update(lifetime_metrics(pred, t.cycles, t.truth, t.start_cycle,
                                    PROTOCOL["eol_threshold"], eol_true=t.eol_true))
        if qpreds[i] is not None:
            rec.update({k: v for k, v in probabilistic_metrics(
                qpreds[i], t.truth_smooth, levels).items()
                if k not in ("q_lo", "q_hi")})
        records.append(rec)
    df = pd.DataFrame(records)
    # 严格按列判断，避免 df.get("error", 空Series) 造成索引错位（原版 bug）
    if "error" in df.columns:
        ok_mask = df["error"].isna()
    else:
        ok_mask = pd.Series(True, index=df.index)
    return df, ok_mask, elapsed, preds, qpreds


def fmt_eol(summ: dict, n_total: int) -> str:
    n_def = summ.get("eol_defined", {}).get("mean", 0) * n_total
    eol = summ.get("eol_abs_err", {})
    hit = summ.get("eol_hit", {}).get("mean", float("nan"))
    if n_def == 0:
        return f"真值 EOL 不可用（0/{n_total}，该库未衰减到 80%）"
    if eol:
        return f"{eol['median']:.0f} 循环 (n={eol['n']})，命中率 {hit:.0%}"
    return f"未触及 EOL（命中率 {hit:.0%}，真值可用 {n_def:.0f}/{n_total}）"


def _preimport_torch(names: List[str]) -> None:
    """兜底：若模块顶部因 argv 判断漏掉，这里再补一次（正常情况下不会走到）。

    真正的修复在文件顶部的 import 段——必须在 import pandas 之前导入 torch。
    """
    if not set(names) & set(_TORCH_MODELS):
        return
    if "torch" in sys.modules:
        return
    try:
        import torch  # noqa: F401
    except ImportError:
        pass


def main():
    args = parse_args()
    if args.setting != "zero":
        raise SystemExit("few-shot 微调尚未实现（方案 W2 阶段）；当前只支持 --setting zero。"
                         "刻意不静默回退到零样本，避免结论口径混淆。")

    names = resolve_model_names(args)
    _preimport_torch(names)
    levels = resolve_levels(args)

    split = None if (args.dataset != "mit" or args.split == "all") else args.split
    print(f"[1/3] 加载数据 {args.dataset} (split={split or 'all'}) ...", flush=True)
    check_dataset(args.dataset)
    cells, meta = load_dataset(args.dataset, DATASETS, split=split, with_meta=True)
    if not cells:
        raise SystemExit(f"数据为空：{args.dataset}，请检查路径 {DATASETS[args.dataset]}")
    if args.max_cells:
        cells = dict(list(cells.items())[:args.max_cells])
    print(f"      {len(cells)} cells，循环数 {min(map(len, cells.values()))}–{max(map(len, cells.values()))}")

    tasks = build_tasks(cells, meta, args.start_fracs, args.quiet)
    print(f"      {len(tasks)} 个评测任务（cell × 起点）")

    suffix = f"-{args.tag}" if args.tag else ""
    out_root = Path(args.output_dir) if args.output_dir else (
        RESULTS_ROOT / f"{args.dataset}-{args.setting}-{args.models or args.model}{suffix}")
    out_root.mkdir(parents=True, exist_ok=True)

    all_summary, model_meta = {}, {}
    for name in names:
        # --seed 只对基础模型生效（基线是确定性算法，不接受 seed 参数）
        extra = {}
        if args.seed is not None:
            from configs.config import MODELS
            if name.lower() in MODELS:
                extra["seed"] = args.seed
        model = get_forecaster(name, **extra)
        if args.max_context:
            # 公平性对照：所有模型看同样长的上下文（TimesFM 上限 512 最紧）
            model.max_context = min(model.max_context or args.max_context, args.max_context)
        print(f"[2/3] 推理：{model.name}"
              + (f"（上下文上限 {model.max_context}）" if args.max_context else ""), flush=True)
        try:
            df, ok_mask, elapsed, preds, qpreds = run_model(
                model, tasks, args.chunk, args.quiet, levels)
        except NotInstalledError as e:
            if len(names) > 1:
                print(f"      跳过 {model.name}：{e}")
                model_meta[model.name] = {"skipped": str(e)}
                continue
            raise

        if df.empty:
            print("      !! 无有效结果")
            continue

        # 口径一致性闸门：分位数链路的中位数层必须逐位等于点预测链路。
        # 不等 ⇒ 两条轨迹已经分叉，CRPS/PICP/MPIW 说的不是这个模型的点预测
        # （timesfm 的分位数层错位一格时实测分叉到 10.93 SOH%，见 metrics.median_deviation）。
        dev = median_deviation(preds, qpreds, levels)
        if dev["n_checked"]:
            if dev["n_bad"]:
                print(f"      ⚠️ 分位数中位数层与点预测不一致：{dev['n_bad']}/"
                      f"{dev['n_checked']} 支，最大偏差 {dev['max_abs']:.4g} SOH%"
                      f"\n         两条链路已分叉，概率指标与点指标不可比！"
                      f"请检查该适配器的分位数层映射。")
            else:
                print(f"      口径一致：{dev['n_checked']} 支的 q{dev['median_level']:.2f} "
                      f"与点预测逐位相同")

        tag = model.name
        df.to_csv(out_root / f"per_cell_{tag}.csv", index=False, encoding="utf-8-sig")
        # 预测与真值成对保存，供画图与逐曲线复核（key = cell@frac）
        pack = {}
        for t, p in zip(tasks, preds):
            if p is None:
                continue
            key = f"{t.cell_id}@{t.frac}"
            pack[key] = np.vstack([np.asarray(p, float), t.truth])
            pack[f"{key}|ctx"] = np.asarray(t.context, float)   # 画轨迹图用
            # 循环号 + 起点：EOL 重设计（RUL 相对误差、过冲循环数）要用
            pack[f"{key}|cyc"] = np.asarray(t.cycles, float)
        if pack:
            np.savez_compressed(out_root / f"predictions_{tag}.npz", **pack)
        # 分位数单独一个文件：与点预测解耦，老工具读不到就自动退回点口径
        if any(q is not None for q in qpreds):
            qpack = {}
            for t, q in zip(tasks, qpreds):
                if q is None:
                    continue
                qpack[f"{t.cell_id}@{t.frac}"] = np.asarray(q, dtype=float)
            if qpack:
                qpack["__levels__"] = np.asarray(levels, dtype=float)
                np.savez_compressed(out_root / f"quantiles_{tag}.npz", **qpack)
                n_cross = int(getattr(model, "n_quantile_crossings", 0))
                msg = (f"      分位数已保存 -> quantiles_{tag}.npz（层 {levels}，"
                       f"CRPS/PICP/MPIW 见 per_cell_{tag}.csv）")
                if n_cross:
                    msg += (f"\n      ⚠️ 分位数穿越 {n_cross} 处已做保序修复"
                            f"（以模型中位数为锚，中位数逐位不变，点预测不受影响）")
                if not args.quiet:
                    print(msg, flush=True)

        ok = df[ok_mask]
        model_meta[tag] = model.info()
        model_meta[tag].update({"elapsed_sec": round(elapsed, 2),
                                "n_ok": int(len(ok)), "n_err": int((~ok_mask).sum()),
                                "chunk_used": min(args.chunk, model.max_horizon or args.chunk)})
        if dev["n_checked"]:
            model_meta[tag]["median_deviation"] = dev
        for frac in sorted(df["start_frac"].unique()):
            sub = ok[ok["start_frac"] == frac]
            if sub.empty:
                continue
            summ = summarize(sub.drop(columns=["cell_id", "start_frac"], errors="ignore")
                             .to_dict("records"))
            all_summary[f"{tag}@{frac}"] = summ
            if not args.quiet:
                mae = summ.get("mae_smooth", {}).get("median", float("nan"))
                print(f"      frac={frac:.2f} | MAE_smooth 中位 {mae:.3f} SOH% | "
                      f"EOL {fmt_eol(summ, len(sub))} | {elapsed:.1f}s / {len(sub)} cells")

    run_meta = {
        "models": names, "dataset": args.dataset, "setting": args.setting,
        "split": split, "start_fracs": args.start_fracs, "chunk": args.chunk,
        "max_context_override": args.max_context or None,
        "seed_override": args.seed,
        "quantile_levels": levels or None,
        "protocol": PROTOCOL, "n_cells": len(cells), "n_tasks": len(tasks),
        "model_info": model_meta, "packages": package_versions(),
        "python": platform.python_version(), "platform": platform.platform(),
        "hf_endpoint": __import__("os").environ.get("HF_ENDPOINT"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    # 已有 summary.json 时**合并**而不是覆盖：模型重、一次跑不完时可按批补齐。
    # （本机后台任务会被会话回收，只能分批在前台跑，因此这个能力是必需的。）
    sum_path = out_root / "summary.json"
    if sum_path.exists():
        try:
            old = json.loads(sum_path.read_text(encoding="utf-8"))
            old_sum = old.get("summary") or {}
            old_meta = (old.get("meta") or {}).get("model_info") or {}
            old_models = (old.get("meta") or {}).get("models") or []
            # 按 `模型@起点` 这个完整键去重，不能只按模型名：
            # 敏感性实验会分多次、每次只跑一个 frac，按模型名去重会把前一次的 frac 抹掉
            # （CALCE 先跑 0.4 再跑 0.6，结果 summary 里只剩 0.6）。
            kept = {k: v for k, v in old_sum.items() if k not in all_summary}
            if kept:
                print(f"      合并已有 summary.json：保留 {len(kept)} 条本次未重算的记录")
            all_summary = {**kept, **all_summary}
            model_meta = {**{k: v for k, v in old_meta.items() if k not in model_meta},
                          **model_meta}
            run_meta["models"] = sorted({*old_models, *names},
                                        key=lambda m: names.index(m) if m in names else 99)
            run_meta["model_info"] = model_meta   # 重新绑定，否则仍指向旧 dict
        except Exception as e:
            print(f"      !! 旧 summary.json 读取失败（{e}），将整体覆盖")

    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump({"meta": run_meta, "summary": all_summary}, f,
                  ensure_ascii=False, indent=2)
    print(f"[3/3] 完成 -> {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
