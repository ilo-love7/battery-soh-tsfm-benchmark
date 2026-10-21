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
    """naive_linear：对上下文做最小二乘线性拟合，再外推 horizon 步。

    **只用上下文最近 ⌊n/2⌋ 个点拟合**（`n // 2`，向下取整）：电池序列的开头往往
    是一段容量平台，把它纳入拟合会把斜率压得接近 0，外推反而跟不上真实衰减。
    这是本文报告的 naive_linear 的真实定义（"线性外推"），不是全段最小二乘。

    ⚠️ `k = n // 2` 这个写法**不能改成** `round(n*0.5)`：奇数长度上下文下两者差 1 个点，
    拟合出的斜率就不同（实测 NASA B0005：k=41 → 84.253，k=42 → 84.339），
    已发表的全部 naive_linear 数字都由 `n // 2` 产生。改这里等于换基线。
    """

    name = "naive_linear"
    fit_frac = 0.5          # 仅作文档标注；实际取 ⌊n*fit_frac⌋ 个点

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        n = len(context)
        if n < 2:
            return np.full(horizon, float(context[-1]))
        x = np.arange(n, dtype=float)
        k = max(2, n // 2)
        b, a = np.polyfit(x[-k:], context[-k:], 1)
        xf = np.arange(n, n + horizon, dtype=float)
        return a + b * xf


class ExpDecay(Forecaster):
    """Q(n) = a·exp(b·n) + c —— 电池文献标准参数化。

    ⚠️ 拟合失败时**允许**退化为线性外推（比整格报废好），但必须留痕：
    每次退化都会 `self.n_fallback += 1` 并发出 RuntimeWarning，
    且 `n_fallback` 会经 `info()` 写进 summary.json。
    此前这里是"注释说记录在案、实际什么都没记"——正是本项目红线
    「绝不静默降级」要防的那类事。已发表的 exp_decay 结果**一次都没触发**
    （已用 predictions_*.npz 逐一核对，无一条与 naive_linear 相同）。
    """

    name = "exp_decay"

    def __init__(self):
        self.n_fallback = 0

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
        except Exception as e:
            self.n_fallback += 1
            warnings.warn(
                f"exp_decay: curve_fit 失败（{type(e).__name__}: {e}），"
                f"本次退化为线性外推（第 {self.n_fallback} 次）",
                RuntimeWarning, stacklevel=2)
            return NaiveLinear().predict(context, horizon)


class ARIMA(Forecaster):
    """statsmodels ARIMA(2,1,2)。未安装 statsmodels 时显式报错。

    ⚠️ 拟合失败退化为 naive_last 时同样留痕（`n_fallback` 进 summary.json）。
    这条路径比 exp_decay 更危险：ARIMA 在噪声大的 SOH 序列上收斂失败并不罕见，
    一旦静默发生，主表里那一行其实是"平推最后一个值"而标签仍写 arima。
    已发表的 arima 结果**一次都没触发**（43 条预测逐一核对，无平推签名）。
    """

    name = "arima"

    def __init__(self, order=(2, 1, 2)):
        self.order = order
        self.n_fallback = 0

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
        except Exception as e:
            self.n_fallback += 1
            warnings.warn(
                f"arima{self.order}: 拟合失败（{type(e).__name__}: {e}），"
                f"本次退化为 naive_last（第 {self.n_fallback} 次）",
                RuntimeWarning, stacklevel=2)
            return NaiveLast().predict(context, horizon)


def all_baselines() -> list[Forecaster]:
    return [NaiveLast(), NaiveLinear(), ExpDecay(), ARIMA()]
