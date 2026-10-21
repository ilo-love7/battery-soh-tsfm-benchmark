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

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        return self.predict_batch([context], horizon)[0]
