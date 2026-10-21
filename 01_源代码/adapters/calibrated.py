"""2 参数单调校准头（对应执行记录 §9.3）。

动机
----
机制分析发现：全量 few-shot 微调带来的收益，主要不是"模型学到了新东西"，
而是**模型敢不敢外推**——微调后的模型愿意把衰减趋势推得更远，
零样本模型则在起点附近"贴着不放"（MIT 上表现为系统性高估 SOH）。

如果机理真是这样，就不必动 190 MB 的权重：只要在**冻结**的基础模型外面
套一个每 chunk 的仿射校准，把"敢不敢外推"这一个自由度显式建模即可：

        p' = c + s · (p − c) + b

    c = 当前上下文最后一个点（模型唯一必对的点，也是唯一的可信锚）
    p = 冻结基础模型本轮预测
    s = 模型预测的"每步变化量"的缩放系数
    b = 每步额外恒定漂移

两个参数的递归语义（为什么不是 3 个）
--------------------------------------
设模型完全不动（p = c），则 p' = c + b，下一轮锚变成 c + b，再预测一次
还是 c + b，于是 p' = c + 2b……轨迹以斜率 b 线性漂移。
所以 b 的语义是"每步补多少衰减"，s 的语义是"模型说的每步变化量打几折"。
两者都是 per-step 量，再引入第三个加性参数（例如给锚加偏移 a）在递归下
与 b **不可辨识**（都表现为每步漂移），因此这个头就是 2 个可辨识参数。

单调性与分位数
--------------
s > 0 时变换严格单调递增，分位数序天然保持，所以同一组 (s, b) 可以直接
用在分位数链路上（区间随 s 一起伸缩），不需要另拟一套参数。
s ≤ 0 会反转分位数序，构造时直接拒绝。

拟合与评测的边界
----------------
- 拟合只用 MIT **train**（batch1，41 支）——与 few-shot 微调同一批数据、
  同一协议（起点 0.5、上下文 512、chunk 64、savgol(11,2) 真值）；
- 评测在 MIT **test** / NASA / CALCE 上，与零样本、全量微调三方对比；
- 拟合目标用 MAE_smooth 的**中位数**（跨 cell），与主表口径一致。
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from adapters.base import DEFAULT_QUANTILE_LEVELS, Forecaster

__all__ = ["CalibratedForecaster", "calibrate"]


def calibrate(pred: np.ndarray, anchor: np.ndarray, s: float, b: float) -> np.ndarray:
    """p' = c + s·(p − c) + b，anchor 与 pred 广播兼容。"""
    return np.asarray(anchor, dtype=float) + s * (np.asarray(pred, dtype=float) -
                                                  np.asarray(anchor, dtype=float)) + b


class CalibratedForecaster(Forecaster):
    """冻结基础模型 + 逐 chunk 仿射校准。

    参数顺序与 `calibrate()` 一致；`s` 必须 > 0。
    支持批量化与分位数（透传给 base，单调变换保序）。
    """

    def __init__(self, base: Forecaster, s: float = 1.0, b: float = 0.0,
                 name: str | None = None):
        s = float(s)
        if not (s > 0):
            raise ValueError(
                f"s 必须 > 0，否则分位数序会反转（当前 s={s}）。"
                f"这个头是单调校准，单调性是不可妥协的。")
        self.base = base
        self.s = s
        self.b = float(b)
        self.name = name or f"{base.name}+cal(s={s:g},b={float(b):g})"
        self.supports_batch = base.supports_batch
        self.cal_params = {"s": self.s, "b": self.b}

    # ------------------------------------------------------------ 上限下传
    # ⚠️ 必须下传到 base：run_benchmark 会 `model.max_context = 512` 做公平对照，
    #    若不透传，包装器的上限是 512、被包装的 Chronos 仍是 2048 —— 校准跑用更长的
    #    上下文，与零样本基线就不在同口径上，对比作废（实测 MIT 上因此差了 0.2 SOH%）。
    @property
    def max_context(self):
        return self.base.max_context

    @max_context.setter
    def max_context(self, v):
        self.base.max_context = v

    @property
    def max_horizon(self):
        return self.base.max_horizon

    @max_horizon.setter
    def max_horizon(self, v):
        self.base.max_horizon = v

    # ------------------------------------------------------------ 锚
    @staticmethod
    def _anchors(contexts: Sequence[np.ndarray]) -> np.ndarray:
        """每个上下文取最后一个点作为锚。

        ⚠️ 不能先 `_truncate` 再取锚：截断取的是**最近** N 点，最后一个点
        不变，所以取不截断的最后一个点等价且更省。真正传给 base 的上下文
        由 base 自己截断。
        """
        return np.array([float(np.asarray(c, dtype=float)[-1]) for c in contexts],
                        dtype=float)

    # ------------------------------------------------------------ 点预测
    def predict_batch(self, contexts: Sequence[np.ndarray],
                      horizon: int) -> List[np.ndarray]:
        p = np.asarray(self.base.predict_batch(contexts, horizon), dtype=float)
        c = self._anchors(contexts)
        out = calibrate(p, c[:, None], self.s, self.b)
        return [out[i] for i in range(out.shape[0])]

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        return self.predict_batch([context], horizon)[0]

    # ------------------------------------------------------------ 分位数
    def supports_quantiles(self) -> bool:
        return self.base.supports_quantiles()

    def predict_quantiles_batch(self, contexts: Sequence[np.ndarray], horizon: int,
                                levels: Sequence[float] = DEFAULT_QUANTILE_LEVELS
                                ) -> np.ndarray:
        """返回 (B, Q, H)。s > 0 ⇒ 单调 ⇒ 分位数序保持，无需再修。"""
        q = np.asarray(
            self.base.predict_quantiles_batch(contexts, horizon, levels), dtype=float)
        if q.ndim != 3:
            raise RuntimeError(f"{self.base.name}: 分位数张量形状异常 {q.shape}")
        c = self._anchors(contexts)                      # (B,)
        # q 是 (B, Q, H)，锚要广播到 (B, 1, 1)——写成 (1, B, 1) 会和 B 轴对不上
        return calibrate(q, c[:, None, None], self.s, self.b)

    # ------------------------------------------------------------ 元数据
    def info(self) -> Dict[str, Any]:
        d = self.base.info()
        d["name"] = self.name
        d["calibrated"] = True
        d["cal_params"] = dict(self.cal_params)
        d["cal_formula"] = "p' = c + s*(p - c) + b"
        d["base_model"] = self.base.name
        return d
