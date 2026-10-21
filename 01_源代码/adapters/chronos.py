"""Chronos / Chronos-Bolt 适配器（Amazon）。

安装：pip install chronos-forecasting（会带 transformers / accelerate / torch）
权重：首次运行从 HF 拉取（走 configs.config 设置的 hf-mirror 镜像）
    chronos-bolt-small  190.9 MB  上下文 2048  patch 结构，直接出分位数
    chronos-t5-small    ~200 MB   上下文 512   采样式，取样本中位数

两个易错点（本文件已处理）：
1. 上下文超长必须截断到**最近** max_context 点，否则 Bolt 直接报错 / T5 静默取前 512 点；
2. Bolt 的 predict_quantiles 返回 (batch, num_quantiles, horizon)，分位数顺序按
   传入的 quantile_levels，不能硬编码索引 4 —— 这里显式传 [0.1, 0.5, 0.9] 并取中间。
"""
from __future__ import annotations

import os
from typing import List, Sequence

import numpy as np

from adapters.base import Forecaster, NotInstalledError

# 各权重名对应的上下文上限（官方 config 里的 context_length）
_CONTEXT_LIMIT = {
    "amazon/chronos-bolt-tiny": 2048,
    "amazon/chronos-bolt-mini": 2048,
    "amazon/chronos-bolt-small": 2048,
    "amazon/chronos-bolt-base": 2048,
    "amazon/chronos-t5-tiny": 512,
    "amazon/chronos-t5-mini": 512,
    "amazon/chronos-t5-small": 512,
    "amazon/chronos-t5-base": 512,
    "amazon/chronos-t5-large": 512,
}


class ChronosForecaster(Forecaster):
    supports_batch = True
    max_horizon = 64          # 官方限制：单次 prediction_length ≤ 64，长 horizon 必须递归

    def __init__(self, bolt: bool = True, model_id: str | None = None,
                 device: str = "cpu", num_samples: int = 20, seed: int = 42,
                 quantile_levels: Sequence[float] = (0.1, 0.5, 0.9)):
        self.bolt = bolt
        self.device = device
        self.num_samples = num_samples
        self.seed = seed
        self.quantile_levels = list(quantile_levels)
        self.name = "chronos-bolt" if bolt else "chronos"
        # few-shot 微调产物通过环境变量注入，避免给 run_benchmark 加一堆专用参数：
        #   CHRONOS_MODEL_ID=amazon/chronos-bolt-small-ftall python run_benchmark.py ...
        # 注意优先级：环境变量 > 配置表里的 model_id。configs.MODELS 已经把
        # model_id 写死了，若按 model_id or env 的顺序，env 永远不会生效
        # （第一版就踩了这个坑，微调后的模型静默加载了原始权重，MAE 一模一样）。
        env_id = os.environ.get("CHRONOS_MODEL_ID") if bolt else os.environ.get("CHRONOS_T5_MODEL_ID")
        self.model_id = env_id or model_id or (
            "amazon/chronos-bolt-small" if bolt else "amazon/chronos-t5-small"
        )
        self.max_context = _CONTEXT_LIMIT.get(self.model_id, 2048 if bolt else 512)
        self._pipe = None

    # ------------------------------------------------------------ 加载
    def _load(self):
        if self._pipe is not None:
            return
        try:
            import torch
            from chronos import BaseChronosPipeline
        except ImportError as e:
            raise NotInstalledError(
                "缺少 chronos-forecasting：pip install chronos-forecasting torch"
            ) from e

        torch.manual_seed(self.seed)
        # CPU 上用 float32；bf16 在 CPU 上反而慢且精度差
        self._torch = torch
        from configs.config import resolve_model_ref
        self.resolved = resolve_model_ref(self.model_id)     # 本地副本优先
        self._pipe = BaseChronosPipeline.from_pretrained(
            self.resolved, device_map=self.device, torch_dtype=torch.float32
        )
        inner = getattr(self._pipe, "model", None)           # 管道是包装器，没有 eval()
        if inner is not None and hasattr(inner, "eval"):
            inner.eval()

    def _to_batch(self, contexts: Sequence[np.ndarray]):
        torch = self._torch
        return [torch.tensor(self._truncate(c), dtype=torch.float32) for c in contexts]

    # ------------------------------------------------------------ 预测
    def predict_quantiles_batch(self, contexts: Sequence[np.ndarray], horizon: int):
        """返回 (B, 3, H) 的 [q10, q50, q90]。"""
        self._load()
        inputs = self._to_batch(contexts)

        if self.bolt:
            q, _ = self._pipe.predict_quantiles(
                inputs, prediction_length=horizon,
                quantile_levels=self.quantile_levels,
            )
            q = q.cpu().numpy().astype(float)
            # 2.3.x 返回 (B, H, Q)，老版本返回 (B, Q, H)——按形状归一，别硬编码
            nq = len(self.quantile_levels)
            if q.ndim == 2:                                  # 单条输入被压成 2 维
                q = q[None, ...]
            if q.shape[1] != nq and q.shape[2] == nq:
                q = np.transpose(q, (0, 2, 1))
            if q.shape[1] != nq:
                raise RuntimeError(f"无法识别 Chronos 分位数张量形状 {q.shape}")
            return q

        # T5：先采样再取分位（与官方 predict_quantiles 口径一致）
        self._torch.manual_seed(self.seed)
        samples = self._pipe.predict(
            inputs, prediction_length=horizon, num_samples=self.num_samples,
            limit_prediction_length=False,
        ).cpu().numpy().astype(float)                    # (B, S, H)
        qs = np.quantile(samples, self.quantile_levels, axis=1)   # (3, B, H)
        return np.transpose(qs, (1, 0, 2))

    def predict_batch(self, contexts: Sequence[np.ndarray], horizon: int) -> List[np.ndarray]:
        q = self.predict_quantiles_batch(contexts, horizon)
        return [q[i, 1, :] for i in range(q.shape[0])]

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        return self.predict_batch([context], horizon)[0]
