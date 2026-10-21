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

    def _quantile_layout(self) -> tuple[np.ndarray, np.ndarray, int]:
        """模型分位数张量最后一维的语义：(层值, 对应列下标, 中位数列)。

        ⚠️ 这里踩过一个大坑。timesfm 2.5 的 `forecast()` 返回
        `(point, quantile)`，其中 `quantile` 的最后一维是
        **`len(config.quantiles) + 1` = 10 列**，不是 9 列。官方
        `timesfm_2p5_torch.py` 里写得很清楚：

            self.q = len(self.config.quantiles) + 1   # 10
            self.aridx = self.config.decode_index     # 5
            ...
            return full_forecast[..., 5], full_forecast   # 点预测就是第 5 列

        也就是说 **第 0 列是一个空槽**——它既不参与 `fix_quantile_crossing`
        （官方只动 1..4 和 6..9），也不被连续分位数头覆写（同上），
        `force_flip_invariance` 的翻转函数还特意把它排除在外。
        真正的 9 个分位数（0.1…0.9）落在第 **1..9** 列，第 5 列既是 0.5 分位
        也逐位等于点预测。

        早先版本写成 `qm = np.linspace(0.1, 0.9, q.shape[-1])`，把空槽也当成
        0.1，整体错位一格：请求 0.5 时 `np.interp` 在「真 q0.4」和「真 q0.5」
        之间插值，**中位数被换掉**。后果不只是概率指标偏差——分位数递归会把
        这个假中位数拼回上下文，于是分位数链路跑出一条与点预测链路不同的轨迹
        （CALCE 实测最大分叉 10.93 SOH%），CRPS 因此可以低于同口径 MAE
        （12/14 支，最大 -3.08），直接违反 CRPS ≥ |y-q50| 的理论下界。
        """
        inner = getattr(self._model, "model", None)
        cfg = getattr(inner, "config", None)
        raw_lv = getattr(cfg, "quantiles", None)
        if not raw_lv:
            raise RuntimeError(
                "无法从 timesfm 模型对象取到分位数层定义（config.quantiles）；"
                "不同版本布局不同，不猜")
        lv = np.asarray([float(x) for x in raw_lv], dtype=float)
        q_cols = getattr(inner, "q", None)
        mid_col = getattr(inner, "aridx", None)

        if q_cols is not None and int(q_cols) == len(lv) + 1:
            cols = np.arange(1, len(lv) + 1)      # 跳过第 0 列空槽
        elif q_cols is not None and int(q_cols) == len(lv):
            cols = np.arange(len(lv))             # 老布局：没有空槽
        else:
            raise RuntimeError(
                f"timesfm 分位数列数 {q_cols} 与层数 {len(lv)} 的布局无法识别")

        # 中位数列必须与 aridx 对齐，否则说明布局假设失效
        mi_lv = int(np.argmin(np.abs(lv - 0.5)))
        if mid_col is None:
            mid_col = int(cols[mi_lv])
        if abs(lv[mi_lv] - 0.5) < 1e-12 and int(mid_col) != int(cols[mi_lv]):
            raise RuntimeError(
                f"中位数列不一致：aridx={mid_col} vs 层映射={int(cols[mi_lv])}")
        return lv, cols, int(mid_col)

    def predict_quantiles_batch(self, contexts: Sequence[np.ndarray], horizon: int,
                                levels: Sequence[float] = (0.1, 0.5, 0.9)):
        """从 TimesFM 的连续分位数头取 (B, Q, H)。

        2.5 的 forecast 返回 (point, quantile)，quantile 形状 (B, H, 10)，
        **分位数层是模型固定的**（0.1…0.9，落在第 1..9 列）。做法：先按
        `_quantile_layout()` 取出真正有效的列，再用线性插值到请求的 levels。
        插值只在**层间**做，不改变模型输出的任何数值；请求层与模型层重合时
        直接取原列，保证 q50 与点预测**逐位相同**（递归拼回上下文才不会分叉）。
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
        if q.ndim == 2:                                     # 单条输入被压成 2 维
            q = q[None, ...]
        lv, cols, _ = self._quantile_layout()
        if q.shape[-1] != int(cols[-1]) + 1:
            # 形状对不上时宁可明确失败，也不猜
            raise RuntimeError(
                f"timesfm 分位数形状 {q.shape} 与层布局 {len(lv)}+空槽 不符")

        lv_req = [float(x) for x in levels]
        sub = q[:, :, cols]                                 # (B, H, Q_model_valid)
        # (B, H, Q) -> 逐 (B,H) 沿"层"轴插值到请求层 -> (B, Q_req, H)
        out = np.empty((q.shape[0], len(lv_req), q.shape[1]), dtype=float)
        for b in range(sub.shape[0]):
            for h in range(sub.shape[1]):
                out[b, :, h] = np.interp(lv_req, lv, sub[b, h, :])
        # 请求层恰好等于模型层时直接取原列（避免插值的浮点尾差）
        for j, want in enumerate(lv_req):
            k = int(np.argmin(np.abs(lv - want)))
            if abs(lv[k] - want) <= 1e-12:
                out[:, j, :] = sub[:, :, k]
        return out

    def predict(self, context: np.ndarray, horizon: int) -> np.ndarray:
        return self.predict_batch([context], horizon)[0]
