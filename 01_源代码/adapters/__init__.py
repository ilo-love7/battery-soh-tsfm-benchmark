"""时序基础模型适配器注册表。

每个适配器都把模型包装成 Forecaster 接口（context → horizon 点预测，SOH 空间）。
依赖未安装时抛 NotInstalledError 并给出安装命令——**绝不静默返回随机数**。

模型名 -> 适配器参数的映射集中在 configs.config.MODELS，避免同一权重名在
多个文件里各写一遍（v1 就是因此把 Chronos 写成 Chromo、把两个模型并成一个）。
"""
from __future__ import annotations

from adapters.base import (Forecaster, NotInstalledError, rolling_forecast,
                           rolling_forecast_batch)

__all__ = ["Forecaster", "NotInstalledError", "get_forecaster",
           "rolling_forecast", "rolling_forecast_batch",
           "list_models", "BASELINES"]

BASELINES = ["naive_last", "naive_linear", "exp_decay", "arima"]


def list_models() -> dict:
    from configs.config import MODELS
    return {"foundation": list(MODELS), "baselines": BASELINES}


def get_forecaster(name: str, **kwargs) -> Forecaster:
    from configs.config import MODELS

    name = name.lower()

    if name in MODELS:
        spec = MODELS[name]
        params = dict(spec.get("kwargs") or {})
        params.update(kwargs)                       # 命令行显式参数优先
        kind = spec["adapter"]
        if kind == "chronos":
            from adapters.chronos import ChronosForecaster
            return ChronosForecaster(**params)
        if kind == "timesfm":
            from adapters.timesfm import TimesFMForecaster
            return TimesFMForecaster(**params)
        if kind == "moirai":
            from adapters.moirai import MoiraiForecaster
            return MoiraiForecaster(**params)
        if kind == "lagllama":
            from adapters.lagllama import LagLlamaForecaster
            return LagLlamaForecaster(**params)
        raise ValueError(f"MODELS[{name}] 的 adapter 字段未知: {kind}")

    # 基线
    from baselines.classic import ARIMA, ExpDecay, NaiveLast, NaiveLinear
    table = {
        "naive_last": NaiveLast,
        "naive_linear": NaiveLinear,
        "exp_decay": ExpDecay,
        "arima": ARIMA,
    }
    if name in table:
        return table[name](**kwargs)
    raise ValueError(f"未知模型: {name}（可选：{list(MODELS)} / {BASELINES}）")
