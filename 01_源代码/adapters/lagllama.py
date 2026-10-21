"""Lag-Llama 适配器（社区，Llama 架构）。

安装（版本敏感，必须独立环境）：
    git clone https://github.com/time-series-foundation-models/lag-llama
    cd lag-llama && pip install -e .
    pip install "gluonts[torch]==0.13.7"
权重：lag-llama.ckpt 从 HuggingFace 手动下载（time-series-foundation-models/Lag-Llama）

说明：走 gluonts 的 estimator/predictor 流程，输出 student-t 分布，取 0.5 分位（中位数）。
本适配器优先级最低（方案 W2-b），装不上就三模型出基准并在论文注明。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np

from adapters.base import Forecaster, NotInstalledError
from configs.config import MODELS_DIR


def _default_ckpt() -> str:
    """权重默认放 shared/models/lag-llama/lag-llama.ckpt；找不到再回落当前目录。"""
    local = MODELS_DIR / "lag-llama" / "lag-llama.ckpt"
    return str(local) if local.exists() else "lag-llama.ckpt"


class LagLlamaForecaster(Forecaster):
    supports_batch = True
    max_horizon = 64

    def __init__(self, ckpt_path: str | None = None, context_length: int | None = None,
                 num_samples: int = 10, batch_size: int = 64, device: str = "cpu",
                 seed: int = 0):
        self.ckpt_path = ckpt_path or _default_ckpt()
        # 官方 ckpt 的 context_length = 32（Lag-Llama 就是短上下文训练的）。
        # 早期版本这里硬编码 512，既比原生慢 ~16 倍，又把模型推到训练分布之外。
        # 传 None 表示"跟随 ckpt 原生值"，在 _load() 里解析。
        self.context_length = context_length
        self.max_context = context_length
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.device = device
        # ⚠️ Lag-Llama 是采样模型：不固定种子时同一份数据重跑结果会变
        # （实测 NASA 上 MAE 7.73 vs 8.70），基准数字就不可复现。
        self.seed = seed
        self._calls = 0
        self.name = "lag-llama"
        self._predictor = None

    def _load(self):
        if self._predictor is not None:
            return
        try:
            import torch
            from lag_llama.gluon.estimator import LagLlamaEstimator
        except ImportError as e:
            raise NotInstalledError(
                "缺少 lag-llama / gluonts：见本文件头部安装说明"
            ) from e

        ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        hp = ckpt["hyper_parameters"]

        # torch>=2.6 的 load_from_checkpoint 默认 weights_only=True，
        # 而官方 ckpt 的 model_kwargs 里存着 gluonts 的分布对象，会被拦下：
        #   UnpicklingError: Unsupported global: StudentTOutput
        # 这里显式放行（权重来自官方仓库，可信）。
        try:
            from gluonts.torch.distributions import (
                StudentTOutput, NegativeBinomialOutput, ImplicitQuantileNetworkOutput,
            )
            torch.serialization.add_safe_globals(
                [StudentTOutput, NegativeBinomialOutput, ImplicitQuantileNetworkOutput]
            )
        except Exception:
            pass

        # 官方 ckpt 的架构参数在 hyper_parameters["model_kwargs"] 里（不在顶层）。
        # 老版本教程写成 hp["n_layer"]，对这份 ckpt 会 KeyError。
        mk = hp.get("model_kwargs") or hp

        # 上下文长度：显式传入优先，否则用 ckpt 原生值（实测 32）
        native_ctx = int(mk.get("context_length") or 32)
        ctx_len = int(self.context_length or native_ctx)
        self.context_length = ctx_len
        self.max_context = ctx_len
        self.native_context = native_ctx

        estimator = LagLlamaEstimator(
            ckpt_path=self.ckpt_path,
            prediction_length=self.max_horizon,
            context_length=ctx_len,
            input_size=mk.get("input_size", 1),
            n_layer=mk["n_layer"],
            n_embd_per_head=mk["n_embd_per_head"],
            n_head=mk["n_head"],
            scaling=mk.get("scaling", "uniform"),
            time_feat=mk.get("time_feat", True),
            batch_size=self.batch_size,
            num_parallel_samples=self.num_samples,   # 注意：不是 num_samples
            device=torch.device(self.device),
        )
        self._predictor = estimator.create_predictor(
            transformation=estimator.create_transformation(),
            module=estimator.create_lightning_module(),
        )

    def _seed_next(self):
        """每次采样前推进一次种子：调用顺序确定 ⇒ 结果可复现，且各 chunk 不复用同一噪声。"""
        import torch
        self._calls += 1
        s = int(self.seed) + self._calls
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        np.random.seed(s)

    def predict_batch(self, contexts: Sequence[np.ndarray], horizon: int) -> List[np.ndarray]:
        self._load()
        from gluonts.dataset.common import ListDataset

        if horizon > self.max_horizon:
            raise ValueError(f"Lag-Llama 单次 horizon 上限 {self.max_horizon}，请减小 rolling_chunk")

        entries = [{"start": "2000-01-01 00:00:00", "target": self._truncate(c)}
                   for c in contexts]
        ds = ListDataset(entries, freq="H")
        self._seed_next()          # 采样前定种子，保证重跑一致
        fcs = list(self._predictor.predict(ds))
        if len(fcs) != len(entries):
            raise RuntimeError(f"Lag-Llama 返回 {len(fcs)} 条预测，期望 {len(entries)}")
        return [np.asarray(fc.quantile(0.5), dtype=float)[:horizon] for fc in fcs]

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        return self.predict_batch([context], horizon)[0]
