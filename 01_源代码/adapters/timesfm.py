"""TimesFM 适配器（Google）。

安装：pip install timesfm[torch]（CPU 可跑）
权重：google/timesfm-2.5-200m-pytorch（走 hf-mirror 镜像）

TimesFM 的 API 在三个大版本里全变过，本适配器按 **2.5 → 2.0 → 1.x** 顺序探测：

    2.5 : TimesFM_2p5_200M_torch.from_pretrained(...) → .compile(ForecastConfig(...))
          → .forecast(horizon=h, inputs=[...]) 返回 (point, quantile)
    2.0 : TimesFm(hparams=TimesFmHparams(...), checkpoint=TimesFmCheckpoint(...))
          → .forecast(inputs, freq) 返回 (point, quantile)
    1.x : TimesFm(context_len=..., horizon_len=..., ...) → .load(ckpt) → .forecast(...)

实际加载后把 model 的 max_context / max_horizon 写回实例属性，供 rolling 夹紧。
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from adapters.base import Forecaster, NotInstalledError

_DEFAULT_REPO = "google/timesfm-2.5-200m-pytorch"


class TimesFMForecaster(Forecaster):
    supports_batch = True
    max_context = 512
    max_horizon = 128

    def __init__(self, checkpoint: str | None = None, backend: str = "cpu",
                 context_len: int | None = None, horizon_len: int = 128):
        self.checkpoint = checkpoint or _DEFAULT_REPO
        self.backend = backend
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.name = "timesfm"
        self.api = "unknown"
        self._model = None

    # ------------------------------------------------------------ 加载
    def _load(self):
        if self._model is not None:
            return
        try:
            import timesfm
        except ImportError as e:
            raise NotInstalledError("缺少 timesfm：pip install 'timesfm[torch]'") from e

        # ---- 2.5 ----
        if hasattr(timesfm, "TimesFM_2p5_200M_torch"):
            from configs.config import resolve_model_ref
            self.resolved = resolve_model_ref(self.checkpoint)   # 本地副本优先
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.resolved)
            ctx = self.context_len or 512
            hz = max(self.horizon_len, 64)
            cfg = timesfm.ForecastConfig(
                max_context=ctx, max_horizon=hz,
                normalize_inputs=True, use_continuous_quantile_head=True,
                force_flip_invariance=True, infer_is_positive=True,
                fix_quantile_crossing=True,
            )
            model.compile(cfg)
            self._model, self.api = model, "2.5"
            self.max_context, self.max_horizon = ctx, hz
            return

        # ---- 2.0 ----
        if hasattr(timesfm, "TimesFmHparams"):
            import os
            from configs.config import resolve_model_ref
            self.resolved = resolve_model_ref(self.checkpoint)
            hparams = timesfm.TimesFmHparams(
                backend=self.backend, per_core_batch_size=1,
                horizon_len=self.horizon_len, context_len=self.context_len or 512,
            )
            ckpt = (timesfm.TimesFmCheckpoint(path=self.resolved)
                    if os.path.isdir(self.resolved)
                    else timesfm.TimesFmCheckpoint(huggingface_repo_id=self.checkpoint))
            model = timesfm.TimesFm(hparams=hparams, checkpoint=ckpt)
            self._model, self.api = model, "2.0"
            self.max_context = self.context_len or 512
            self.max_horizon = self.horizon_len
            return

        # ---- 1.x ----
        model = timesfm.TimesFm(
            context_len=self.context_len or 512,
            horizon_len=self.horizon_len,
            input_patch_len=32, output_patch_len=128,
            num_layers=20, model_dims=1280, backend=self.backend,
        )
        model.load_from_checkpoint(repo_id=self.checkpoint)
        self._model, self.api = model, "1.x"
        self.max_context = self.context_len or 512
        self.max_horizon = self.horizon_len

    # ------------------------------------------------------------ 预测
    def predict_batch(self, contexts: Sequence[np.ndarray], horizon: int) -> List[np.ndarray]:
        self._load()
        inputs = [self._truncate(c) for c in contexts]

        if self.api == "2.5":
            point, _ = self._model.forecast(horizon=horizon, inputs=inputs)
        else:
            # 2.0 / 1.x：位置参数 (inputs, freq)，freq=0 表示未知频次
            try:
                point, _ = self._model.forecast(inputs, freq=[0] * len(inputs))
            except TypeError:
                point, _ = self._model.forecast(inputs, freq=0)

        arr = np.asarray(point, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        return [arr[i, :horizon] for i in range(arr.shape[0])]

    def supports_quantiles(self) -> bool:
        # 2.5 有连续分位数头；2.0/1.x 的第二个返回值在部分版本里是 None，
        # 所以统一在 predict_quantiles_batch 里判空，不支持就明确抛错。
        self._load()
        return self.api == "2.5"

    def predict_quantiles_batch(self, contexts: Sequence[np.ndarray], horizon: int,
                                levels: Sequence[float] = (0.1, 0.5, 0.9)):
        """从 TimesFM 的连续分位数头取 (B, Q, H)。

        2.5 的 forecast 返回 (point, quantile)，quantile 形状 (B, H, Q) 且
        **分位数层是模型固定的**（不是我们传的 levels）。做法：先取模型自己的
        分位数层，再用线性插值到请求的 levels。插值只在**层间**做，
        不改变模型输出的任何数值。
        """
        self._load()
        if self.api != "2.5":
            raise NotImplementedError(
                f"timesfm API {self.api} 不返回可用的分位数（第二返回值为 None）")
        inputs = [self._truncate(c) for c in contexts]
        point, quant = self._model.forecast(horizon=horizon, inputs=inputs)
        if quant is None:
            raise NotImplementedError("timesfm 2.5 未返回分位数张量")

        q = np.asarray(quant, dtype=float)                  # (B, H, Q_model)
        lv_req = [float(x) for x in levels]
        # 模型固定的分位数层（timesfm 2.5 的连续头默认十分位数）
        qm = getattr(self._model, "quantile_levels", None)
        if qm is None:
            qm = np.linspace(0.1, 0.9, q.shape[-1])
        qm = np.asarray(qm, dtype=float)
        if q.ndim == 2:                                     # 单条输入被压成 2 维
            q = q[None, ...]
        if q.shape[-1] != len(qm):
            # 形状对不上时宁可明确失败，也不猜
            raise RuntimeError(f"timesfm 分位数形状 {q.shape} 与层数 {len(qm)} 不符")

        # (B, H, Q_model) -> 逐 (B,H) 插值到请求层 -> (B, Q_req, H)
        out = np.empty((q.shape[0], len(lv_req), q.shape[1]), dtype=float)
        for b in range(q.shape[0]):
            # q[b] 是 (H, Q_model)：对每个 horizon 步，沿"层"轴插值到请求层
            out[b] = np.stack([np.interp(lv_req, qm, q[b, h, :])
                               for h in range(q.shape[1])], axis=1)
        return out

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        return self.predict_batch([context], horizon)[0]
