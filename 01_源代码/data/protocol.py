"""评测协议：清洗、SOH 归一、相对起点切分、平滑参考。

协议要点（与方案 §3/§6 一致，所有模型与基线共用同一套）：
1. 清洗：排序、去重、剔除 NaN/非正值、只保留最长连续段、剔除离群点；
2. SOH 归一：q_ref = 上下文前 5 点中位数（因果，无测试集泄漏），SOH = cap/q_ref*100；
3. 切分：相对起点 start = floor(frac * L)，context = 前段，horizon = 后段；
4. 平滑参考：Savitzky-Golay(11, 2)，只用于评估，不喂模型。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


@dataclass
class Split:
    context: np.ndarray        # SOH 上下文（%）
    truth: np.ndarray          # SOH 真值后段（%）
    truth_raw: np.ndarray      # 原始容量后段（Ah）
    truth_smooth: np.ndarray   # 平滑后真值（%），评估主参考
    start_cycle: int           # 预测起点循环号
    cycles: np.ndarray         # 后段循环号
    q_ref: float               # SOH 归一化基准（Ah）
    eol_true: Optional[float] = None  # 官方 EOL（MIT cycle_life）；None 时按 80% 穿越判定


def clean_series(df: pd.DataFrame, outlier_thresh: float = 0.03) -> pd.DataFrame:
    """排序去重 + 只保留最长连续段 + 剔除离群点。"""
    df = df.dropna(subset=["capacity_Ah"]).copy()
    df = df[df["capacity_Ah"] > 0]
    df = df.sort_values("cycle").drop_duplicates("cycle")

    if df.empty:
        return df

    # 最长连续段（cycle 差 == 1）
    c = df["cycle"].to_numpy()
    breaks = np.where(np.diff(c) != 1)[0]
    if len(breaks):
        seg_id = np.concatenate([[0], np.cumsum(np.diff(c) != 1)])
        sizes = np.bincount(seg_id)
        df = df[seg_id == sizes.argmax()]

    # 离群点：与前后 5 点滚动中位数的相对偏差
    if len(df) >= 11:
        med = df["capacity_Ah"].rolling(5, center=True, min_periods=3).median()
        rel = (df["capacity_Ah"] - med).abs() / med
        df = df[rel <= outlier_thresh]
    return df.reset_index(drop=True)


def soh_normalize(cap: np.ndarray, ref_points: int = 5) -> Tuple[np.ndarray, float]:
    """按序列自身前 ref_points 点中位数归一为 SOH(%)。"""
    q_ref = float(np.median(cap[:min(ref_points, len(cap))]))
    if q_ref <= 0:
        raise ValueError("归一化基准容量非正")
    return cap / q_ref * 100.0, q_ref


def relative_split(df: pd.DataFrame, frac: float = 0.5,
                   smooth_window: int = 11, smooth_order: int = 2,
                   ref_points: int = 5,
                   eol_true: Optional[float] = None) -> Optional[Split]:
    """相对起点切分。序列过短（<20 循环）返回 None。

    eol_true: 官方 EOL 循环号（MIT 用 cycle_life）。数据在 EOL 处截断时，
    靠"SOH≤80%"永远检测不到真值 EOL，必须显式传入。
    """
    if len(df) < 20:
        return None
    L = len(df)
    start = int(np.floor(frac * L))
    start = max(start, 10)                     # 上下文至少 10 点
    if L - start < 5:
        return None

    cap = df["capacity_Ah"].to_numpy(dtype=float)
    ctx_raw, hor_raw = cap[:start], cap[start:]

    # SOH 归一化基准只用上下文（因果）
    soh_ctx, q_ref = soh_normalize(ctx_raw, ref_points)
    soh_hor = hor_raw / q_ref * 100.0

    # 平滑参考（真值序列整体平滑后再切，避免边界效应）
    if L >= smooth_window:
        cap_s = savgol_filter(cap, smooth_window, smooth_order)
    else:
        cap_s = cap
    truth_smooth = cap_s[start:] / q_ref * 100.0

    return Split(
        context=soh_ctx,
        truth=soh_hor,
        truth_raw=hor_raw,
        truth_smooth=truth_smooth,
        start_cycle=int(df["cycle"].iloc[start]),
        cycles=df["cycle"].to_numpy()[start:],
        q_ref=q_ref,
        eol_true=eol_true,
    )


def eol_cycle(soh: np.ndarray, cycles: np.ndarray, threshold: float = 80.0) -> Optional[float]:
    """SOH 首次 ≤ threshold 的循环号（线性插值）。未到阈值返回 None。"""
    below = np.where(soh <= threshold)[0]
    if len(below) == 0:
        return None
    i = below[0]
    if i == 0:
        return float(cycles[0])
    x0, x1 = soh[i - 1], soh[i]
    c0, c1 = cycles[i - 1], cycles[i]
    if x0 == x1:
        return float(c1)
    t = (x0 - threshold) / (x0 - x1)
    return float(c0 + t * (c1 - c0))
