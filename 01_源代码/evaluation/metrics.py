"""评估指标（全部在 SOH% 空间）。

轨迹层：MAE / RMSE（主指标用平滑真值，副指标用原始点真值）
寿命层：EOL 绝对误差（循环数）、RUL@起点误差
聚合：中位数 + 均值 + IQR（Severson 论文口径 = 中位误差）
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from data.protocol import eol_cycle


def trajectory_metrics(pred: np.ndarray, truth: np.ndarray,
                       truth_smooth: Optional[np.ndarray] = None) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    n = min(len(pred), len(truth))
    pred, truth = pred[:n], truth[:n]

    mae = float(np.mean(np.abs(pred - truth)))
    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    # MAPE 只作参考（接近 EOL 时分母小，会爆炸）
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.nanmean(np.abs((truth - pred) / truth)) * 100)

    out = {"mae_raw": mae, "rmse_raw": rmse, "mape_raw": mape}
    if truth_smooth is not None:
        ts = np.asarray(truth_smooth, dtype=float)[:n]
        out["mae_smooth"] = float(np.mean(np.abs(pred - ts)))
        out["rmse_smooth"] = float(np.sqrt(np.mean((pred - ts) ** 2)))
    return out


def lifetime_metrics(pred: np.ndarray, cycles: np.ndarray, truth: np.ndarray,
                     start_cycle: int, threshold: float = 80.0,
                     eol_true: Optional[float] = None) -> Dict[str, float]:
    """EOL 误差与 RUL 误差。

    eol_true: 官方 EOL（MIT 用 cycle_life）。MIT 序列在 EOL 处截断，
              按穿越判定永远找不到，必须显式给真值。
    真值未到 EOL 且无官方标签的 cell 不参与 EOL 统计。
    """
    cycles = np.asarray(cycles)
    truth_arr = np.asarray(truth, float)
    out: Dict[str, float] = {}
    if eol_true is None:
        # ⚠️ 前提闸门：起点本身已跌破阈值时，"首次穿越"会退化到第 0 个点，
        # EOL 预测变得无意义（CALCE 的 CX2 系列在 50% 起点处已到 ~79.5%）。
        # 这类 cell 不计入 EOL 统计，否则会出现"命中率 100%、误差 0 循环"的假象。
        if truth_arr.size and truth_arr[0] <= threshold:
            out["eol_defined"] = 0.0
            out["eol_start_below"] = 1.0
            return out
        eol_true = eol_cycle(truth_arr, cycles, threshold)
    if eol_true is None:
        out["eol_defined"] = 0.0
        return out

    # 预测 EOL：把预测序列拼到起点后的循环号上
    pred_cycles = cycles[:len(pred)]
    eol_pred = eol_cycle(np.asarray(pred, float), pred_cycles, threshold)
    out["eol_defined"] = 1.0
    out["eol_true"] = float(eol_true)
    if eol_pred is None:
        # 预测未触及 EOL：按"超出观测范围"处理，记误差为 NaN 并计数
        out["eol_hit"] = 0.0
        out["eol_pred"] = float("nan")
        out["eol_abs_err"] = float("nan")
        out["rul_abs_err"] = float("nan")
        return out

    out["eol_hit"] = 1.0
    out["eol_pred"] = float(eol_pred)
    out["eol_abs_err"] = abs(eol_pred - eol_true)
    out["rul_abs_err"] = abs((eol_pred - start_cycle) - (eol_true - start_cycle))
    return out


def summarize(records: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """对一组 cell 的结果做中位数/均值/IQR 汇总。"""
    if not records:
        return {}
    keys = set().union(*(r.keys() for r in records))
    out: Dict[str, Dict[str, float]] = {}
    for k in sorted(keys):
        vals = np.array([r[k] for r in records if k in r and np.isfinite(r[k])], dtype=float)
        if vals.size == 0:
            continue
        out[k] = {
            "n": int(vals.size),
            "median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "q25": float(np.percentile(vals, 25)),
            "q75": float(np.percentile(vals, 75)),
            "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
        }
    return out
