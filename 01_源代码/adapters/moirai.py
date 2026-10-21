"""Moirai 适配器（Salesforce，uni2ts）。

安装：pip install uni2ts（会带 gluonts / lightning / torch）
权重：Salesforce/moirai-1.0-R-small（走 hf-mirror 镜像）

说明：
- uni2ts 基于 gluonts 的 Predictor 接口；电池"每循环一个点"没有真实时间频次，
  按协议统一传 freq='H' 占位（只影响时间特征，不影响点预测口径）；
- 输出为分位数张量，取 0.5 分位；
- 批量：一次把多个 cell 放进同一个 ListDataset，Predictor 内部按 batch_size 前向。
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from adapters.base import Forecaster, NotInstalledError


class MoiraiForecaster(Forecaster):
    supports_batch = True
    max_horizon = 64

    def __init__(self, model_id: str = "Salesforce/moirai-1.0-R-small",
                 context_length: int = 512, patch_size: str = "auto",
                 num_samples: int = 100, batch_size: int = 32,
                 device: str = "cpu", seed: int = 0):
        self.model_id = model_id
        self.context_length = context_length
        self.max_context = context_length
        self.patch_size = patch_size
        self.num_samples = num_samples
        # ⚠️ Moirai 用采样解码（num_samples=100 取中位）：不固定种子时重跑结果会漂移
        # （实测 NASA 上 7.110 vs 7.092）。固定种子后才谈得上"可复现基准"。
        self.seed = seed
        self._calls = 0
        self.batch_size = batch_size
        self.device = device
        self.name = "moirai"
        self._predictor = None

    def _load(self):
        if self._predictor is not None:
            return
        try:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        except ImportError as e:
            raise NotInstalledError("缺少 uni2ts：pip install uni2ts") from e

        from configs.config import resolve_model_ref
        self.resolved = resolve_model_ref(self.model_id)     # 本地副本优先
        module = MoiraiModule.from_pretrained(self.resolved)
        self._factory = MoiraiForecast(
            module=module,
            prediction_length=self.max_horizon,
            context_length=self.context_length,
            patch_size=self.patch_size,
            num_samples=self.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self._predictor = self._factory.create_predictor(batch_size=self.batch_size)

    def _seed_next(self):
        """每次采样前推进一次种子：调用顺序确定 ⇒ 结果可复现，且各 chunk 不复用同一噪声。"""
        import torch
        self._calls += 1
        s = int(self.seed) + self._calls
        torch.manual_seed(s)
        np.random.seed(s)

    def predict_batch(self, contexts: Sequence[np.ndarray], horizon: int) -> List[np.ndarray]:
        self._load()
        from gluonts.dataset.common import ListDataset

        if horizon > self.max_horizon:
            raise ValueError(f"Moirai 单次 horizon 上限 {self.max_horizon}，请减小 rolling_chunk")

        entries = [{"start": "2000-01-01 00:00:00", "target": self._truncate(c)}
                   for c in contexts]
        ds = ListDataset(entries, freq="H")
        self._seed_next()
        fcs = list(self._predictor.predict(ds))
        if len(fcs) != len(entries):
            raise RuntimeError(f"Moirai 返回 {len(fcs)} 条预测，期望 {len(entries)}")
        return [np.asarray(fc.quantile(0.5), dtype=float)[:horizon] for fc in fcs]

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        return self.predict_batch([context], horizon)[0]
