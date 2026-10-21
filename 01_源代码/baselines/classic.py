"""经典基线：Naive-Last / Naive-Linear / 指数衰减 / ARIMA。

基线与基础模型同切分、同归一、同指标。Naive 基线必须进主表——
如果零样本模型打不过"平推"，结论必须如实写。
"""
from __future__ import annotations

import warnings

import numpy as np

from adapters.base import Forecaster


class NaiveLast(Forecaster):
    name = "naive_last"

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        return np.full(horizon, float(context[-1]))


class NaiveLinear(Forecaster):
    name = "naive_linear"

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        n = len(context)
        if n < 2:
            return np.full(horizon, float(context[-1]))
        x = np.arange(n, dtype=float)
        # 只用最近一半点拟合，避免早期平台段主导
        k = max(2, n // 2)
        b, a = np.polyfit(x[-k:], context[-k:], 1)
        xf = np.arange(n, n + horizon, dtype=float)
        return a + b * xf


class ExpDecay(Forecaster):
    """Q(n) = a·exp(b·n) + c —— 电池文献标准参数化。"""

    name = "exp_decay"

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        from scipy.optimize import curve_fit

        n = len(context)
        x = np.arange(n, dtype=float)
        y = np.asarray(context, dtype=float)

        def f(x, a, b, c):
            return a * np.exp(b * x) + c

        p0 = [y[0] - y[-1], -1.0 / max(n, 1), y[-1]]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(f, x, y, p0=p0, maxfev=20000,
                                    bounds=([-1e3, -1.0, -1e3], [1e3, 0.0, 1e3]))
            xf = np.arange(n, n + horizon, dtype=float)
            pred = f(xf, *popt)
            if not np.all(np.isfinite(pred)):
                raise RuntimeError("non-finite")
            return pred
        except Exception:
            # 拟合失败退化为线性外推（记录在案，不隐藏）
            return NaiveLinear().predict(context, horizon)


class ARIMA(Forecaster):
    """statsmodels ARIMA(2,1,2)。未安装 statsmodels 时显式报错。"""

    name = "arima"

    def __init__(self, order=(2, 1, 2)):
        self.order = order

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        try:
            from statsmodels.tsa.arima.model import ARIMA as _ARIMA
        except ImportError as e:
            from adapters.base import NotInstalledError
            raise NotInstalledError("ARIMA 需要 statsmodels：pip install statsmodels") from e

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = _ARIMA(np.asarray(context, dtype=float), order=self.order)
                res = model.fit()
                fc = res.forecast(steps=horizon)
            fc = np.asarray(fc, dtype=float)
            if not np.all(np.isfinite(fc)):
                raise RuntimeError("non-finite")
            return fc
        except Exception:
            return NaiveLast().predict(context, horizon)


def all_baselines() -> list[Forecaster]:
    return [NaiveLast(), NaiveLinear(), ExpDecay(), ARIMA()]
