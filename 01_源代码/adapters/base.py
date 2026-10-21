"""统一预测器接口。

所有模型（基础模型适配器 / 经典基线）实现同一接口：

    class Forecaster:
        name: str
        def predict(self, context: np.ndarray, horizon: int) -> np.ndarray

context 为 SOH(%) 一维序列，返回长度 = horizon 的点预测（SOH%）。
递归滚动由 rolling_forecast() / rolling_forecast_batch() 统一处理，保证各模型调用方式一致。

批量接口 predict_batch() 默认退化为逐条循环；支持批的模型（Chronos / Moirai）
覆写它，能把 43 个 cell 一次前向跑完——CPU 上这是主要加速来源。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence

import numpy as np


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

    def info(self) -> Dict[str, Any]:
        """元数据（写进 summary.json，满足验收标准 #5）。"""
        d: Dict[str, Any] = {"name": self.name, "max_context": self.max_context,
                             "max_horizon": self.max_horizon,
                             "supports_batch": self.supports_batch}
        for k in ("model_id", "resolved", "checkpoint", "device", "num_samples",
                  "seed", "bolt", "api", "order", "fit_frac", "ref_points",
                  "n_fallback"):
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


class NotInstalledError(RuntimeError):
    """模型依赖未安装。绝不静默降级为随机数。"""
