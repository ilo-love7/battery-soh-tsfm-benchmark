"""统一预测器接口。

所有模型（基础模型适配器 / 经典基线）实现同一接口：

    class Forecaster:
        name: str
        def predict(self, context: np.ndarray, horizon: int) -> np.ndarray

context 为 SOH(%) 一维序列，返回长度 = horizon 的点预测（SOH%）。
递归滚动由 rolling_forecast() / rolling_forecast_batch() 统一处理，保证各模型调用方式一致。

批量接口 predict_batch() 默认退化为逐条循环；支持批的模型（Chronos / Moirai）
覆写它，能把 43 个 cell 一次前向跑完——CPU 上这是主要加速来源。

分位数接口（2026-10-21 新增，对应执行记录 §9.1）
-----------------------------------------------
基础模型原生输出**预测分布**（Chronos-Bolt 直接出分位数、TimesFM 有连续分位数头、
Moirai / Lag-Llama 是采样解码），但 v1 基准只取了中位数点预测，
等于把模型最核心的优势扔掉了。新增一套并行接口：

    supports_quantiles() -> bool
    predict_quantiles_batch(contexts, horizon, levels) -> (B, Q, H)
    rolling_forecast_quantiles(model, context, horizon, chunk, levels) -> (Q, H)

⚠️ 递归时**把中位数拼回上下文**，与 rolling_forecast 的点预测口径完全一致。
   否则同一个模型在两条链路上的中位数轨迹会不同，概率指标与点指标就不可比。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence

import numpy as np

# 默认分位数层：0.1/0.5/0.9 → 名义 80% 预测区间
DEFAULT_QUANTILE_LEVELS = (0.1, 0.5, 0.9)


class Forecaster(ABC):
    name: str = "base"
    # 单次可接受的最长上下文；None = 不限制。超长时由基类截断（取最近 N 点）
    max_context: int | None = None
    # 是否支持批量前向
    supports_batch: bool = False
    # 单次前向可输出的最长 horizon；None = 不限制。rolling 里 chunk 会被夹到它以内
    max_horizon: int | None = None

    @abstractmethod
    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        """单次预测。context: (n,) SOH；返回 (horizon,) SOH。"""

    def predict_batch(self, contexts: Sequence[np.ndarray], horizon: int) -> List[np.ndarray]:
        """批量预测。默认逐条调用 predict()，支持批的模型覆写。"""
        return [self.predict(c, horizon) for c in contexts]

    def predict_series(self, context: np.ndarray, horizon: int, chunk: int = 64) -> np.ndarray:
        """滚动递归预测：每次预测 chunk 步，拼回上下文，直到覆盖 horizon。

        这是协议强制要求（方案 §6.2）：避免长 horizon 一次硬预测导致误差发散。
        """
        return rolling_forecast(self, context, horizon, chunk)

    # -------------------------------------------------------- 随机状态
    def rng_checkpoint(self):
        """采样型模型：返回一个可恢复的随机状态标记；确定性模型返回 None。

        为什么需要这一对钩子
        --------------------
        采样型适配器（Moirai / Lag-Llama）用**全局单调的调用计数器**推种子
        （`_calls += 1; seed = base + _calls`），这样"重跑一致"且各 chunk 不复用
        同一噪声。但点预测链路和分位数链路是**两遍独立遍历**：

            第 1 遍 rolling_forecast_batch         → 用掉计数器 1..N
            第 2 遍 rolling_forecast_quantiles_batch → 接着用 N+1..2N

        同一个 chunk 在两遍里拿到的是**不同种子、不同样本**，于是
        `q50 ≠ 点预测`，分位数递归沿着另一条轨迹跑下去。实测（未修复时）：

            moirai      CALCE max|p−q50| = 0.75 SOH%，MIT 4.52，NASA 5.76
            lag-llama   CALCE max|p−q50| = 62.5 SOH%，MIT 78.2，NASA 45.2

        做法：`run_benchmark` 在每组的点预测链路**之前**取一次 checkpoint，
        在分位数链路**之前**恢复它。这样两遍的每个 chunk 用同一种子、采到同一批
        样本 ⇒ `q50` 与点预测逐位相同。点预测链路的调用全部发生在恢复之前，
        **主表数字一位都不变**；恢复后分位数链路跑完，计数器恰好回到原值，
        后续分组的点预测链路也不受影响。
        """
        return None

    def rng_restore(self, ckpt) -> None:
        """把随机状态恢复回 `rng_checkpoint()` 的返回值。确定性模型无需实现。"""
        return None

    # ------------------------------------------------------------ 分位数
    def supports_quantiles(self) -> bool:
        """是否原生输出预测分布。基线一律 False（点估计没有区间可言）。"""
        return False

    def predict_quantiles_batch(self, contexts: Sequence[np.ndarray], horizon: int,
                                levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS
                                ) -> np.ndarray:
        """批量分位数预测，返回 (B, Q, H)。

        levels 必须**升序**且含中位数层（用于递归时拼回上下文）。
        不支持的模型抛 NotImplementedError——绝不返回伪造的区间。
        """
        raise NotImplementedError(f"{self.name} 不支持分位数输出")

    def predict_quantiles(self, context: np.ndarray, horizon: int,
                          levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS) -> np.ndarray:
        """单条分位数预测，返回 (Q, H)。"""
        return self.predict_quantiles_batch([context], horizon, levels)[0]

    def info(self) -> Dict[str, Any]:
        """元数据（写进 summary.json，满足验收标准 #5）。"""
        d: Dict[str, Any] = {"name": self.name, "max_context": self.max_context,
                             "max_horizon": self.max_horizon,
                             "supports_batch": self.supports_batch,
                             "supports_quantiles": self.supports_quantiles()}
        for k in ("model_id", "resolved", "checkpoint", "device", "num_samples",
                  "seed", "bolt", "api", "order", "fit_frac", "ref_points",
                  "n_fallback", "quantile_levels", "n_quantile_crossings"):
            if hasattr(self, k):
                d[k] = getattr(self, k)
        return d

    def _truncate(self, ctx: np.ndarray) -> np.ndarray:
        """超长上下文取**最近** N 点（不是最前 N 点）。"""
        ctx = np.asarray(ctx, dtype=float)
        if self.max_context is not None and len(ctx) > self.max_context:
            return ctx[-self.max_context:]
        return ctx


def rolling_forecast(model: Forecaster, context: np.ndarray, horizon: int,
                     chunk: int = 64) -> np.ndarray:
    if model.max_horizon:
        chunk = min(chunk, model.max_horizon)
    ctx = np.asarray(context, dtype=float).copy()
    out = []
    remaining = horizon
    while remaining > 0:
        step = min(chunk, remaining)
        pred = np.asarray(model.predict(ctx, step), dtype=float)
        if pred.shape != (step,):
            raise ValueError(f"{model.name}: 预测长度 {pred.shape} != {step}")
        out.append(pred)
        ctx = np.concatenate([ctx, pred])
        remaining -= step
    return np.concatenate(out)


def _median_index(levels: Sequence[float]) -> int:
    """levels 里最接近 0.5 的下标；不含 0.5 时取中间那个。"""
    lv = [float(x) for x in levels]
    if 0.5 in lv:
        return lv.index(0.5)
    return int(np.argmin([abs(x - 0.5) for x in lv]))


def quantiles_from_forecast(fc, levels: Sequence[float], horizon: int) -> np.ndarray:
    """从 gluonts 的 `SampleForecast` 取 (Q, horizon) 分位数。

    ⚠️ 必须用 `fc.quantile()`，**不能**用 `np.quantile(fc.samples, ...)`。
    两者对"分位数"的定义不同：

        gluonts  SampleForecast.quantile:  sample_idx = round((S−1)·q)
                                          return self._sorted_samples[sample_idx]
                                          → 取**最近样本**，不做插值
        numpy   np.quantile(..., 0.5):    线性插值，取第 49/50 个次序统计量的均值

    同一批样本下，两者差**半个样本间距**（实测 0.0157 SOH%，S=100）。
    点预测链路走的是 `fc.quantile(0.5)`（见 `predict_batch`），所以分位数链路
    一旦改用 `np.quantile`，`q50` 就与点预测对不上——分位数递归会沿另一条轨迹跑，
    moirai 首 chunk 差 0.062、lag-llama 差 4.38 SOH%，全段放大到 32。

    ⚠️ 时间轴是**最后一维**，且 horizon 切片要在 `fc.quantile` 之后做——
    与 `predict_batch` 里 `fc.quantile(0.5)[:horizon]` 完全同构。
    历史坑：曾写成 `samples[:horizon]`（切的是**样本轴**），
    horizon < num_samples 时不报错但只用了前 horizon 条样本；
    horizon >= num_samples 时直接广播失败。
    """
    lv = [float(x) for x in levels]
    rows = [np.asarray(fc.quantile(l), dtype=float)[:horizon] for l in lv]
    return np.vstack(rows)


def repair_quantiles(model: Forecaster, q: np.ndarray,
                     levels: Sequence[float]) -> np.ndarray:
    """修复"分位数穿越"（quantile crossing），**以中位数为锚**，并计数。

    ⚠️ 为什么要修而不是报错：Chronos-Bolt 在 MIT 43 支的完整递归里，
    263 个 chunk 中只有 1 个出现穿越，共 5/31066 个点（0.016%），
    量级约 0.02 SOH%，且集中在最后一个不满 64 步的小 chunk 上。
    为这 5 个点让整轮概率分析失败，是把数值瑕疵当成数据问题——
    PICP/MPIW/CRPS 需要非负区间宽度才有意义。

    ⚠️ 为什么不能简单 `np.sort`：实测踩过。排序会**换掉模型的 q50**——
    例如模型输出 (q10, q50, q90) = (95.951, 95.939, 95.953)，
    排序后变成 (95.939, 95.951, 95.953)，中位数从 95.939 变成 95.951。
    于是分位数递归拼回上下文的"中位数"与点预测链路差 0.012 SOH%，
    `predictions_*.npz` 与已发表版本不再逐位一致（实测复现了这个偏差）。
    排序取中间值在数学上是三者中位数，但**不是模型声明的中位数**。

    现在的做法：以 `mi`（模型的中位数层）为锚做保序修复——
        下侧: min(cummax(q[:mi]),  q[mi])   非减且不超过锚
        上侧: max(cummax(q[mi+1:]), q[mi])  非减且不小于锚
    结果保证单调不减，且 `q[mi]` **逐位不变**。因此：
      · 点预测链路与已发表数字完全不受影响；
      · 分位数链路拼回的也是同一个中位数，两条链路严格一致。
    """
    q = np.asarray(q, dtype=float).copy()
    if q.ndim != 2:
        return q
    if not np.any(np.diff(q, axis=0) < 0):
        return q
    model.n_quantile_crossings = int(getattr(model, "n_quantile_crossings", 0)) + \
        int(np.sum(np.diff(q, axis=0) < 0))
    mi = _median_index(levels)
    anchor = q[mi]
    if mi > 0:
        q[:mi] = np.minimum(np.maximum.accumulate(q[:mi], axis=0), anchor)
    if mi + 1 < q.shape[0]:
        q[mi + 1:] = np.maximum(np.maximum.accumulate(q[mi + 1:], axis=0), anchor)
    return q


def rolling_forecast_quantiles(model: Forecaster, context: np.ndarray, horizon: int,
                               chunk: int = 64,
                               levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS
                               ) -> np.ndarray:
    """滚动递归分位数预测，返回 (Q, horizon)。

    ⚠️ 与 rolling_forecast 的关键约定：递归时**把中位数拼回上下文**，
    不是把某个分位数或整个分布拼回去。这样两条链路的中位数轨迹逐位相同，
    概率指标（CRPS/PICP）才能和点指标（MAE）放在同一张表里比较。
    """
    if model.max_horizon:
        chunk = min(chunk, model.max_horizon)
    lv = [float(x) for x in levels]
    mi = _median_index(lv)
    ctx = np.asarray(context, dtype=float).copy()
    out: List[np.ndarray] = []
    remaining = horizon
    while remaining > 0:
        step = min(chunk, remaining)
        q = np.asarray(model.predict_quantiles(ctx, step, lv), dtype=float)
        if q.ndim != 2 or q.shape[1] != step:
            raise ValueError(f"{model.name}: 分位数形状 {q.shape} != ({len(lv)}, {step})")
        if q.shape[0] != len(lv):
            raise ValueError(f"{model.name}: 分位数层数 {q.shape[0]} != {len(lv)}")
        q = repair_quantiles(model, q, lv)
        # 修复后必须单调不减；真不成立说明修复逻辑有 bug
        if np.any(np.diff(q, axis=0) < 0):
            raise ValueError(f"{model.name}: 修复后仍不单调（实现 bug）")
        out.append(q)
        ctx = np.concatenate([ctx, q[mi]])       # 只拼中位数
        remaining -= step
    return np.concatenate(out, axis=1)


def rolling_forecast_batch(model: Forecaster, contexts: Sequence[np.ndarray],
                           horizons: Sequence[int], chunk: int = 64) -> List[np.ndarray]:
    """批量滚动递归：每轮把所有 cell 当前上下文一次性送进模型。

    比逐 cell 循环快 5–20 倍（CPU 上批量前向的摊薄效应明显），
    且各 cell 的 chunk 轮数不同，用 active 列表逐步淘汰已完成的 cell。
    """
    if not model.supports_batch:
        return [rolling_forecast(model, c, h, chunk) for c, h in zip(contexts, horizons)]

    if model.max_horizon:
        chunk = min(chunk, model.max_horizon)
    ctxs = [np.asarray(c, dtype=float).copy() for c in contexts]
    outs: List[List[np.ndarray]] = [[] for _ in ctxs]
    done = [0] * len(ctxs)
    active = [i for i, h in enumerate(horizons) if h > 0]

    while active:
        steps = [min(chunk, horizons[i] - done[i]) for i in active]
        # 同一轮内各 cell 步长可能不同：按步长分组，保证一次前向内 horizon 一致
        by_step: Dict[int, List[int]] = {}
        for i, s in zip(active, steps):
            by_step.setdefault(s, []).append(i)

        for step, idxs in by_step.items():
            preds = model.predict_batch([ctxs[i] for i in idxs], step)
            for i, p in zip(idxs, preds):
                p = np.asarray(p, dtype=float)
                if p.shape != (step,):
                    raise ValueError(f"{model.name}: 批量预测长度 {p.shape} != {step}")
                outs[i].append(p)
                ctxs[i] = np.concatenate([ctxs[i], p])
                done[i] += step

        active = [i for i in active if done[i] < horizons[i]]

    return [np.concatenate(o) if o else np.empty(0) for o in outs]


def rolling_forecast_quantiles_batch(model: Forecaster, contexts: Sequence[np.ndarray],
                                     horizons: Sequence[int], chunk: int = 64,
                                     levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS
                                     ) -> List[np.ndarray]:
    """批量滚动分位数预测，返回 List[(Q, H_i)]。

    与 rolling_forecast_batch 同构：按步长分组、active 列表淘汰。
    不支持批的模型退化为逐条 rolling_forecast_quantiles。
    """
    lv = [float(x) for x in levels]
    if not model.supports_batch:
        return [rolling_forecast_quantiles(model, c, h, chunk, lv)
                for c, h in zip(contexts, horizons)]
    if model.max_horizon:
        chunk = min(chunk, model.max_horizon)

    ctxs = [np.asarray(c, dtype=float).copy() for c in contexts]
    outs: List[List[np.ndarray]] = [[] for _ in ctxs]
    done = [0] * len(ctxs)
    active = [i for i, h in enumerate(horizons) if h > 0]

    while active:
        steps = [min(chunk, horizons[i] - done[i]) for i in active]
        by_step: Dict[int, List[int]] = {}
        for i, s in zip(active, steps):
            by_step.setdefault(s, []).append(i)

        for step, idxs in by_step.items():
            q = np.asarray(
                model.predict_quantiles_batch([ctxs[i] for i in idxs], step, lv),
                dtype=float)
            if q.ndim != 3:
                raise ValueError(f"{model.name}: 批量分位数形状 {q.shape}，期望 (B,Q,H)")
            if q.shape[0] != len(idxs) or q.shape[2] != step or q.shape[1] != len(lv):
                raise ValueError(f"{model.name}: 批量分位数形状 {q.shape} != "
                                 f"({len(idxs)},{len(lv)},{step})")
            mi = _median_index(lv)
            for j, i in enumerate(idxs):
                qi = repair_quantiles(model, q[j], lv)
                # 修复后必须单调不减；真不成立说明修复逻辑有 bug
                if np.any(np.diff(qi, axis=0) < 0):
                    raise ValueError(f"{model.name}: 修复后仍不单调（实现 bug）")
                outs[i].append(qi)
                ctxs[i] = np.concatenate([ctxs[i], qi[mi]])
                done[i] += step

        active = [i for i in active if done[i] < horizons[i]]

    return [np.concatenate(o, axis=1) if o else np.empty((len(lv), 0)) for o in outs]


class NotInstalledError(RuntimeError):
    """模型依赖未安装。绝不静默降级为随机数。"""
