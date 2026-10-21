"""评估指标（全部在 SOH% 空间）。

轨迹层：MAE / RMSE（主指标用平滑真值，副指标用原始点真值）
概率层：CRPS / PICP / MPIW（分位数口径，对应执行记录 §9.1）
寿命层：EOL 绝对误差（循环数）、RUL 相对误差、EOL 过冲循环数、未穿越裕度
聚合：中位数 + 均值 + IQR（Severson 论文口径 = 中位误差）

概率层存在的理由：基础模型**原生输出预测分布**（Chronos-Bolt 直接出分位数、
TimesFM 有连续分位数头、Moirai / Lag-Llama 是采样解码），v1 基准只取了中位数，
等于把模型最核心的优势扔掉了。这里把它补回来。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from data.protocol import eol_cycle

DEFAULT_LEVELS = (0.1, 0.5, 0.9)


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

    ⚠️ 2026-10-21 审计：原先这里同时输出 `eol_abs_err` 与 `rul_abs_err`，
    而 `rul_abs_err = |(eol_pred-start)-(eol_true-start)|` 与
    `eol_abs_err = |eol_pred-eol_true|` **数学上恒等**，且全仓库没有任何地方读它。
    留着只会让人误以为"寿命层有两个独立指标"。已删除，只保留 eol_abs_err。
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
        # ⚠️ 这条 early-return 正是 MIT 的主路径（数据在 EOL 处截断，零样本模型
        # 几乎不可能在观测窗内跌破 80%）。v1 就在这里把"差多少"整个丢掉了，
        # 于是 MIT 的 EOL 命中率恒为 0%、不含信息量。重设计指标必须覆盖它。
        _add_eol_redesign(out, pred, start_cycle, eol_true, threshold)
        return out

    out["eol_hit"] = 1.0
    out["eol_pred"] = float(eol_pred)
    out["eol_abs_err"] = abs(eol_pred - eol_true)
    _add_eol_redesign(out, pred, start_cycle, eol_true, threshold)
    return out


def _add_eol_redesign(out: Dict[str, float], pred: np.ndarray,
                      start_cycle: int, eol_true: float,
                      threshold: float) -> None:
    """EOL 指标重设计（执行记录 §9.2）——补三个 v1 缺失的量。

    v1 的 EOL 命中率在 MIT 上**恒为 0%**：数据在 EOL 处截断，"预测首次跌破 80%"
    对零样本模型几乎不可能发生，于是这个指标在 MIT 上不含任何信息量。
    但"没跌破"和"差 0.1 个点没跌破"显然是两回事，v1 把它们混成一个 0。

    新增：
    1. `eol_overshoot_cyc`：带符号过冲 = eol_pred − eol_true。
       正 = 预测偏晚（穿越点循环号更大，衰减被低估 = 保守）；
       负 = 预测偏早（穿越点循环号更小，衰减被高估 = 乐观/激进）。
       NaN = 未穿越。
       ⚠️ 符号方向曾经写反过（把"正"说成偏早），已按公式 eol_pred − eol_true 订正。
    2. `rul_rel_err`：|eol_pred − eol_true| / (eol_true − start_cycle)，
       即相对于**起点剩余寿命**的相对误差，比"绝对循环数"跨 cell 可比。
    3. `no_cross_margin`：未穿越时，预测序列末段的 SOH 与阈值的距离
       （正 = 还高于阈值，即过于乐观）。有穿越时为 NaN。
       → 让 MIT 的"0% 命中率"变成"平均还差 X 个 SOH 点"这种可读的数。
    """
    p = np.asarray(pred, dtype=float)
    eol_pred = out.get("eol_pred")
    remaining_life = float(eol_true) - float(start_cycle)

    if eol_pred is not None and np.isfinite(eol_pred):
        over = float(eol_pred) - float(eol_true)
        out["eol_overshoot_cyc"] = over
        out["rul_rel_err"] = abs(over) / remaining_life if remaining_life > 0 else float("nan")
        out["no_cross_margin"] = float("nan")
    else:
        out["eol_overshoot_cyc"] = float("nan")
        out["rul_rel_err"] = float("nan")
        # 未穿越：看末段（最后 5 点或全部）离阈值还有多远
        tail = p[-min(5, len(p)):] if len(p) else np.empty(0)
        out["no_cross_margin"] = float(np.mean(tail) - threshold) if tail.size else float("nan")

    # 真值侧的剩余寿命，供分析脚本做归一（真值本身可能也未穿越）
    out["rul_true_cyc"] = remaining_life


def crps_from_quantiles(y: float, q: np.ndarray, levels: Sequence[float]) -> float:
    """由分位数近似 CRPS（Continuous Ranked Probability Score）。

    公式（Gneiting & Raftery 的积分式在分位数层上的离散化）：

        CRPS(F, y) = ∫₀¹ 2·(QL_α − y)·(1{y ≤ QL_α} − α) dα

    把 [0,1] 按 levels 切成若干段、段内 QL 取该段右端的分位数，得

        CRPS ≈ Σ_{i=1}^{k+1} 2(q_i − y)·[ 1{y ≤ q_i}·(α_i − α_{i−1})
                                        − (α_i² − α_{i−1}²)/2 ]

    其中 α_0 = 0、α_{k+1} = 1，q_{k+1} = q_k（用端点分位数外延）。

    ⚠️ 退化自检：只有一个分位数层且它恰好是中位数时，公式必须退化成 |y − m|。
    这一点有单元测试锁死（tests/test_prob_metrics.py）——分位数近似的常见错误
    就是把退化情形算成 0.25/0.75 的不对称形式。
    """
    lv = [float(x) for x in levels]
    qv = np.asarray(q, dtype=float)
    if qv.shape[0] != len(lv):
        raise ValueError(f"分位数层数 {qv.shape[0]} != levels 数 {len(lv)}")
    # levels 必须升序（调用方保证），这里再兜一次
    if any(b <= a for a, b in zip(lv, lv[1:])):
        raise ValueError(f"levels 必须严格升序：{lv}")

    y = float(y)
    alpha = [0.0] + lv + [1.0]
    qs = [qv[0]] + [qv[i] for i in range(len(lv))] + [qv[-1]]
    total = 0.0
    for i in range(1, len(alpha)):
        a0, a1 = alpha[i - 1], alpha[i]
        qi = qs[i]
        ind = 1.0 if y <= qi else 0.0
        total += 2.0 * (qi - y) * (ind * (a1 - a0) - (a1 * a1 - a0 * a0) / 2.0)
    return float(total)


def probabilistic_metrics(pred_q: np.ndarray, truth: np.ndarray,
                          levels: Sequence[float] = DEFAULT_LEVELS,
                          interval: Optional[tuple] = None) -> Dict[str, float]:
    """概率口径指标。

    pred_q : (Q, H) 分位数预测，levels 升序
    truth  : (H,) 真值（用**平滑真值**，与 MAE 主指标同源）
    levels : 分位数层
    interval: 计算 PICP / MPIW 用的区间层，默认取最外层 (levels[0], levels[-1])

    返回：
        crps      : 全 horizon 的平均 CRPS（SOH%）
        picp      : 预测区间覆盖概率（真值落在 [q_lo, q_hi] 内的比例）
        mpiw      : 平均预测区间宽度（SOH%）
        pinaw     : MPIW 归一化（除以真值极差），跨 cell 可比
        q_lo/q_hi : 实际用的区间层
    """
    q = np.asarray(pred_q, dtype=float)
    t = np.asarray(truth, dtype=float)
    if q.ndim != 2:
        raise ValueError(f"pred_q 应为 (Q,H)，得到 {q.shape}")
    n = min(q.shape[1], len(t))
    q, t = q[:, :n], t[:n]
    if n == 0:
        return {"crps": float("nan"), "picp": float("nan"),
                "mpiw": float("nan"), "pinaw": float("nan")}

    lv = [float(x) for x in levels]
    crps = float(np.mean([crps_from_quantiles(t[j], q[:, j], lv) for j in range(n)]))

    lo_i, hi_i = (0, len(lv) - 1) if interval is None else interval
    lo, hi = q[lo_i], q[hi_i]
    picp = float(np.mean((t >= lo) & (t <= hi)))
    mpiw = float(np.mean(hi - lo))
    rng = float(np.max(t) - np.min(t))
    pinaw = mpiw / rng if rng > 1e-12 else float("nan")
    return {"crps": crps, "picp": picp, "mpiw": mpiw, "pinaw": pinaw,
            "q_lo": lv[lo_i], "q_hi": lv[hi_i]}


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
