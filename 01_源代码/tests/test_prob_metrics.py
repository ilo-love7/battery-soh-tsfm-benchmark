"""概率指标（CRPS / PICP / MPIW）与 EOL 重设计的回归测试。

无第三方依赖（不需要 pytest），直接 `python tests/test_prob_metrics.py` 跑。

锁死的东西
----------
1. **CRPS 退化情形**：只有一个分位数层且它恰好是中位数时，
   CRPS 必须**精确等于** |y − m|。分位数近似的常见错误就是把退化情形算成
   0.25/0.75 的不对称形式——这一条是硬约束。
2. **CRPS 与精确采样 CRPS 一致**：按公式的"分段常数"假设反演样本，
   用精确样本 CRPS（mean|x−y| − 0.5·mean|x−x'|）对照。
   ⚠️ 参照物必须和公式同一个分布：若参照物用"相邻分位数间线性插值"，
   那是另一个分布，对不上是必然的（第一版就栽在这里）。
3. CRPS 非负、且 y 取中位数时最小。
4. PICP / MPIW 语义（含越界、零宽区间、自定义区间层）。
5. EOL 重设计：过冲符号、RUL 相对误差、未穿越裕度、前提闸门未被破坏。
6. 分位数递归**只拼中位数**，保证与点预测链路中位数逐位相同。
7. 不支持分位数的模型必须明确抛 NotImplementedError。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.metrics import (crps_from_quantiles,  # noqa: E402
                                lifetime_metrics,
                                probabilistic_metrics)

_PASS: list = []
_FAIL: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  — {detail}" if detail and not cond else ""))


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# ---------------------------------------------------------------- 1. CRPS 退化
def test_crps_degenerate_equals_abs_error() -> None:
    """唯一分位数层 = 中位数 ⇒ CRPS ≡ |y − m|（硬约束，必须精确）。"""
    print("\n[1] CRPS 退化情形 == |y − m|（精确）")
    m = 88.5
    for y in (80.0, 85.0, 88.5, 90.0, 95.0, 100.0):
        got = crps_from_quantiles(y, np.array([m]), [0.5])
        want = abs(y - m)
        check(f"y={y}: {got:.6f} == {want:.6f}", close(got, want), f"{got} vs {want}")


def test_crps_three_levels_degenerate() -> None:
    """三层分位数但宽度为 0（q10=q50=q90=m）⇒ 仍应等于 |y − m|。"""
    print("\n[2] 三层但零宽度 ⇒ 仍等于 |y − m|（精确）")
    m = 91.0
    for y in (82.0, 91.0, 97.0):
        got = crps_from_quantiles(y, np.array([m, m, m]), [0.1, 0.5, 0.9])
        want = abs(y - m)
        check(f"y={y}: {got:.6f} == {want:.6f}", close(got, want), f"{got} vs {want}")


def test_crps_matches_exact_sampling() -> None:
    """与精确采样 CRPS 对照（同一分布：分段常数分位函数）。"""
    print("\n[3] CRPS 与精确采样 CRPS 一致")
    rng = np.random.default_rng(7)
    n = 40000
    worst = 0.0
    for trial in range(6):
        nl = int(rng.integers(2, 5))
        pool = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        lv = sorted(rng.choice(pool, size=nl, replace=False).tolist())
        qv = np.sort(rng.uniform(70, 100, size=nl))
        y = float(rng.uniform(70, 100))

        analytic = crps_from_quantiles(y, qv, lv)

        # 按公式的分段常数假设反演样本
        alpha = [0.0] + lv + [1.0]
        qs = [qv[0]] + [qv[i] for i in range(len(lv))] + [qv[-1]]
        widths = np.diff(np.array(alpha))
        probs = widths / widths.sum()
        r = np.random.default_rng(100 + trial)
        xs = np.asarray(qs[1:], dtype=float)[r.choice(len(probs), size=n, p=probs)]
        # 精确样本 CRPS
        t1 = float(np.mean(np.abs(xs - y)))
        sub = xs[r.choice(n, size=3000, replace=False)]
        t2 = 0.5 * float(np.mean(np.abs(sub[:, None] - sub[None, :])))
        exact = t1 - t2
        rel = abs(analytic - exact) / max(abs(exact), 1e-9)
        worst = max(worst, rel)
    check(f"6 组随机设置，最大相对差 {worst:.2%} < 6%", worst < 0.06, f"{worst:.2%}")


def test_crps_nonneg_and_min_at_median() -> None:
    print("\n[4] CRPS 非负，且在 y=中位数 处最小")
    q = np.array([86.0, 90.0, 94.0])
    lv = [0.1, 0.5, 0.9]
    vals = [crps_from_quantiles(y, q, lv) for y in (84.0, 88.0, 90.0, 92.0, 96.0)]
    check("全部非负", all(v >= -1e-12 for v in vals), str(vals))
    check("y=中位数(90) 处最小", vals[2] == min(vals),
          f"argmin at {['84', '88', '90', '92', '96'][int(np.argmin(vals))]}")


# ---------------------------------------------------------------- 5. PICP/MPIW
def test_picp_mpiw() -> None:
    print("\n[5] PICP / MPIW 语义")
    lv = [0.1, 0.5, 0.9]
    # (Q=3, H=2)：h=0 区间 [86,94]，h=1 区间 [87,93]
    q = np.array([[86.0, 87.0], [90.0, 90.0], [94.0, 93.0]])
    t = np.array([89.0, 93.5])          # 89 在内，93.5 > 93 在外
    m = probabilistic_metrics(q, t, lv)
    check("PICP == 0.5（一个在内一个在外）", close(m["picp"], 0.5), str(m["picp"]))
    check("MPIW == mean(8, 6) == 7.0", close(m["mpiw"], 7.0), str(m["mpiw"]))

    t_all_in = np.array([89.0, 90.0])
    check("都在内 ⇒ PICP == 1", close(probabilistic_metrics(q, t_all_in, lv)["picp"], 1.0))

    t_all_out = np.array([80.0, 95.0])
    check("都在外 ⇒ PICP == 0", close(probabilistic_metrics(q, t_all_out, lv)["picp"], 0.0))

    # 零宽区间
    q0 = np.array([[90.0, 90.0], [90.0, 90.0], [90.0, 90.0]])
    m3 = probabilistic_metrics(q0, np.array([90.0, 91.0]), lv)
    check("零宽区间 MPIW == 0", close(m3["mpiw"], 0.0), str(m3["mpiw"]))
    check("零宽区间 PICP == 0.5（90 在内、91 在外）", close(m3["picp"], 0.5), str(m3["picp"]))
    # 零宽 ⇒ CRPS 退回 |y−m|，对两个 horizon 求平均
    check("零宽区间 CRPS == mean(0, 1) == 0.5", close(m3["crps"], 0.5), str(m3["crps"]))


def test_custom_interval() -> None:
    print("\n[6] 自定义区间层")
    lv = [0.05, 0.25, 0.5, 0.75, 0.95]
    q = np.array([[80.0, 80.0], [85.0, 85.0], [90.0, 90.0], [95.0, 95.0], [99.0, 99.0]])
    m = probabilistic_metrics(q, np.array([84.0, 96.0]), lv, interval=(0, 4))
    check("interval=(0,4) ⇒ 用 [80,99]", close(m["q_lo"], 0.05) and close(m["q_hi"], 0.95))
    check("两个点都在 [80,99] 内 ⇒ PICP=1", close(m["picp"], 1.0), str(m["picp"]))
    check("MPIW == 19", close(m["mpiw"], 19.0), str(m["mpiw"]))

    m2 = probabilistic_metrics(q, np.array([84.0, 96.0]), lv, interval=(1, 3))
    check("interval=(1,3) ⇒ 用 [85,95]",
          close(m2["q_lo"], 0.25) and close(m2["q_hi"], 0.75))
    check("84<85 且 96>95 ⇒ PICP=0", close(m2["picp"], 0.0), str(m2["picp"]))
    check("MPIW == mean(10, 10) == 10", close(m2["mpiw"], 10.0), str(m2["mpiw"]))


# ---------------------------------------------------------------- 7. EOL 重设计
def test_eol_redesign() -> None:
    """用整数步长的真值，让穿越点可以手算。

    truth[i] = 95 − i  ⇒ 在 i=15 处正好等于 80，eol_true = cycles[15] = 115
    起点 100 ⇒ 剩余寿命 rul_true = 15
    """
    print("\n[7] EOL 重设计三个量")
    thr = 80.0
    cycles = np.arange(100, 200)
    truth = 95.0 - np.arange(100, dtype=float)

    # (a) 预测与真值同步 ⇒ 过冲 0、RUL 相对误差 0
    m = lifetime_metrics(truth.copy(), cycles, truth, 100, thr, eol_true=115.0)
    check("同时穿越：eol_hit == 1", close(m["eol_hit"], 1.0))
    check("同时穿越：eol_pred == 115", close(m["eol_pred"], 115.0), str(m.get("eol_pred")))
    check("同时穿越：过冲 == 0", close(m["eol_overshoot_cyc"], 0.0),
          str(m.get("eol_overshoot_cyc")))
    check("同时穿越：rul_rel_err == 0", close(m["rul_rel_err"], 0.0),
          str(m.get("rul_rel_err")))
    check("rul_true_cyc == 15", close(m["rul_true_cyc"], 15.0))

    # (b) 预测更陡（每步 −2）⇒ 第 7.5 步穿越 ⇒ eol_pred = 107.5
    pred_steep = 95.0 - 2.0 * np.arange(100, dtype=float)
    m2 = lifetime_metrics(pred_steep, cycles, truth, 100, thr, eol_true=115.0)
    check("更陡：eol_pred == 107.5", close(m2["eol_pred"], 107.5), str(m2.get("eol_pred")))
    check("更陡：过冲 == −7.5", close(m2["eol_overshoot_cyc"], -7.5),
          str(m2.get("eol_overshoot_cyc")))
    check("更陡：rul_rel_err == 7.5/15 == 0.5", close(m2["rul_rel_err"], 0.5),
          str(m2.get("rul_rel_err")))
    check("过冲 == eol_pred − eol_true",
          close(m2["eol_overshoot_cyc"], m2["eol_pred"] - m2["eol_true"]))

    # (c) 预测更平（每步 −0.1）⇒ 100 步只降 10 点，永不跌破 ⇒ 走 early-return 分支
    pred_flat_slow = 95.0 - 0.1 * np.arange(100, dtype=float)
    m3 = lifetime_metrics(pred_flat_slow, cycles, truth, 100, thr, eol_true=115.0)
    check("更平且不跌破：末值 85.1 > 80", pred_flat_slow[-1] > thr)
    check("更平且不跌破：eol_hit == 0", close(m3["eol_hit"], 0.0))
    check("不跌破：eol_abs_err 为 NaN", np.isnan(m3["eol_abs_err"]))
    check("不跌破：过冲为 NaN", np.isnan(m3["eol_overshoot_cyc"]))
    check("不跌破：rul_rel_err 为 NaN", np.isnan(m3["rul_rel_err"]))
    check("不跌破：裕度 > 0（过于乐观）", m3["no_cross_margin"] > 0,
          str(m3.get("no_cross_margin")))
    tail_mean = float(np.mean(pred_flat_slow[-5:]))
    check(f"裕度 == 末5点均值({tail_mean:.2f}) − 80 == {tail_mean-80:.2f}",
          close(m3["no_cross_margin"], tail_mean - 80.0, 1e-9),
          str(m3.get("no_cross_margin")))
    check("不跌破时仍记录 rul_true_cyc", close(m3["rul_true_cyc"], 15.0))

    # (d) 完全不跌（常数 85）⇒ 裕度恰为 5
    m4 = lifetime_metrics(np.full(100, 85.0), cycles, truth, 100, thr, eol_true=115.0)
    check("常数 85 ⇒ 裕度 == 5.0", close(m4["no_cross_margin"], 5.0),
          str(m4.get("no_cross_margin")))


def test_eol_gate_still_works() -> None:
    """闸门不能被动过：起点已 ≤80% 的 cell 仍不计入 EOL 统计。"""
    print("\n[8] EOL 前提闸门未被破坏")
    thr = 80.0
    cycles = np.arange(0, 50)
    truth = np.linspace(79.0, 70.0, 50)          # 起点就低于 80%
    m = lifetime_metrics(np.linspace(79, 70, 50), cycles, truth, 0, thr)
    check("起点低于阈值 ⇒ eol_defined == 0", close(m["eol_defined"], 0.0))
    check("起点低于阈值 ⇒ eol_start_below == 1", close(m.get("eol_start_below", 0.0), 1.0))


# ---------------------------------------------------------------- 9. 递归中位数
def test_rolling_quantiles_feeds_median_only() -> None:
    """分位数递归必须只把中位数拼回上下文。"""
    print("\n[9] rolling_forecast_quantiles 只拼中位数")
    from adapters.base import (Forecaster, rolling_forecast,
                               rolling_forecast_quantiles)

    class Fake(Forecaster):
        name = "fake"
        supports_batch = True
        max_horizon = 4

        def predict(self, context, horizon):
            base = float(np.asarray(context)[-1]) - 1.0
            return np.full(horizon, base)

        def predict_quantiles_batch(self, contexts, horizon, levels=(0.1, 0.5, 0.9)):
            out = []
            for c in contexts:
                base = float(np.asarray(c)[-1]) - 1.0
                out.append(np.array([np.full(horizon, base - 2.0),
                                     np.full(horizon, base),
                                     np.full(horizon, base + 2.0)]))
            return np.array(out)

        def supports_quantiles(self):
            return True

    f = Fake()
    ctx = np.array([100.0, 99.0, 98.0])
    p = rolling_forecast(f, ctx, 10, chunk=4)
    q = rolling_forecast_quantiles(f, ctx, 10, chunk=4)
    check("分位数形状 == (3, 10)", q.shape == (3, 10), str(q.shape))
    check("中位数层与点预测逐位相同",
          np.allclose(q[1], p), f"max diff {np.max(np.abs(q[1] - p))}")
    check("q10 <= q50 <= q90 全程成立", np.all(q[0] <= q[1]) and np.all(q[1] <= q[2]))
    # 每 chunk 内是常数（= ctx_end − 1），跨 chunk 边界才递减 1
    check("chunk 内中位数为常数", np.allclose(q[1][:4], 97.0) and np.allclose(q[1][4:8], 96.0))
    check("跨 chunk 边界递减 1",
          close(float(q[1][4] - q[1][3]), -1.0) and close(float(q[1][8] - q[1][7]), -1.0),
          f"{q[1]}")
    check("q10 始终比中位数低 2", np.allclose(q[0], q[1] - 2.0))
    check("q90 始终比中位数高 2", np.allclose(q[2], q[1] + 2.0))


def test_unsupported_model_raises() -> None:
    """不支持分位数的模型必须明确抛错，不能返回伪造区间。"""
    print("\n[10] 不支持分位数 ⇒ 明确抛 NotImplementedError")
    from adapters.base import Forecaster

    class PointOnly(Forecaster):
        name = "point-only"

        def predict(self, context, horizon):
            return np.zeros(horizon)

    f = PointOnly()
    check("supports_quantiles() == False", f.supports_quantiles() is False)
    try:
        f.predict_quantiles(np.zeros(5), 4)
        check("predict_quantiles 抛错", False, "没有抛异常")
    except NotImplementedError:
        check("predict_quantiles 抛 NotImplementedError", True)
    except Exception as e:      # noqa: BLE001
        check("predict_quantiles 抛 NotImplementedError", False,
              f"抛的是 {type(e).__name__}")


def test_baselines_report_no_quantiles() -> None:
    """四个经典基线都应报告 supports_quantiles() == False。"""
    print("\n[11] 经典基线不支持分位数")
    from baselines.classic import ARIMA, ExpDecay, NaiveLast, NaiveLinear
    for cls in (NaiveLast, NaiveLinear, ExpDecay, ARIMA):
        f = cls()
        check(f"{cls.__name__}.supports_quantiles() == False", f.supports_quantiles() is False)
        # info() 里必须带这个字段（否则 summary.json 看不出有没有概率口径）
        check(f"{cls.__name__}.info() 含 supports_quantiles",
              "supports_quantiles" in f.info())


def test_repair_quantiles_pins_median() -> None:
    """分位数穿越修复必须以模型的中位数为锚，且结果单调不减。

    这条锁死一个实测踩过的坑：用 np.sort 修复会把模型的 q50 换掉
    （(95.951, 95.939, 95.953) 排序后中位数变成 95.951），
    于是分位数递归与点预测链路的中位数差 0.012 SOH%，
    predictions_*.npz 与已发表版本不再逐位一致。
    """
    print("\n[12] 穿越修复：锚定中位数 + 单调不减")
    from adapters.base import Forecaster, repair_quantiles

    class M(Forecaster):
        name = "m"

        def predict(self, context, horizon):
            return np.zeros(horizon)

    m = M()
    lv = [0.1, 0.5, 0.9]

    # q10 > q50 的穿越（Chronos-Bolt 实测形态）
    q = np.array([[95.9507, 90.0, 80.0],
                  [95.9390, 90.0, 80.0],
                  [95.9531, 89.0, 79.0]])
    out = repair_quantiles(m, q, lv)
    check("修复后单调不减", np.all(np.diff(out, axis=0) >= 0))
    check("中位数层逐位不变（锚定）",
          np.allclose(out[1], q[1]), f"{out[1]} vs {q[1]}")
    check("q10 被夹到不超过中位数", out[0, 0] <= out[1, 0])
    check("q90 被抬到不小于中位数", out[2, 0] >= out[1, 0])
    check("第 2 列 q50 > q90 也被修好", out[1, 1] <= out[2, 1])
    check("第 3 列整体穿越也修好", np.all(np.diff(out[:, 2]) >= 0))
    check("穿越计数已记录", getattr(m, "n_quantile_crossings", 0) > 0,
          str(getattr(m, "n_quantile_crossings", 0)))

    # 无穿越时必须是恒等映射（逐位不变，不能碰任何数）
    q2 = np.array([[86.0, 87.0], [90.0, 90.0], [94.0, 93.0]])
    m2 = M()
    out2 = repair_quantiles(m2, q2, lv)
    check("无穿越时恒等（逐位相同）", np.array_equal(out2, q2))
    check("无穿越时不计数", getattr(m2, "n_quantile_crossings", 0) == 0)

    # 5 层也要正确：锚在 0.5（索引 2）
    lv5 = [0.05, 0.25, 0.5, 0.75, 0.95]
    q5 = np.array([[92.0, 80.0],
                   [91.0, 80.0],
                   [90.0, 80.0],
                   [89.0, 80.0],
                   [88.0, 80.0]])          # 完全倒序 + 一个常数列
    out5 = repair_quantiles(M(), q5, lv5)
    check("5 层：修复后单调不减", np.all(np.diff(out5, axis=0) >= 0))
    check("5 层：中位数层（索引 2）不变", np.allclose(out5[2], q5[2]))
    check("5 层：常数列修复后全等于 80", np.allclose(out5[:, 1], 80.0))


# ------------------------------------- 13. 采样型分位数的取法必须与点预测一致
def test_quantiles_from_forecast_matches_point() -> None:
    """`fc.quantile()`，不是 `np.quantile(fc.samples, ...)`。

    Moirai / Lag-Llama 真实踩过的两个坑，都在这一层：

    (a) **切错轴**：写成 `samples[:horizon]` 切的是**样本轴**。
        horizon < num_samples 时不报错但只用了前 horizon 条样本；
        horizon >= num_samples 时直接广播失败
        （实测 `could not broadcast (3,64) into (3,21)`）。

    (b) **取法不同**（更隐蔽，直到加了口径闸门才暴露）：gluonts 的
        `SampleForecast.quantile` 是
            `sample_idx = round((S−1)·q); return _sorted_samples[sample_idx]`
        即**取最近样本、不插值**；而 `np.quantile(..., 0.5)` 是线性插值。
        同一批样本下两者差**半个样本间距**（实测 0.0157 SOH%）。
        点预测链路走 `fc.quantile(0.5)`，分位数链路若改用 `np.quantile`，
        `q50` 就对不上 ⇒ 递归沿另一条轨迹跑（moirai 首 chunk 0.062、
        lag-llama 4.38 SOH%，全段放大到 32）。

    这里用一个**模仿 gluonts 语义**的假 forecast 把两条不变量同时锁死。
    """
    print("\n[13] 采样型分位数：取法与点预测一致 + 时间轴切片正确")
    from adapters.base import quantiles_from_forecast

    class FakeSampleForecast:
        """按 gluonts 的真实语义实现 quantile：取最近样本，不做插值。"""

        def __init__(self, samples: np.ndarray) -> None:
            self.samples = np.asarray(samples, dtype=float)
            self.num_samples = self.samples.shape[0]
            self._sorted = np.sort(self.samples, axis=0)

        def quantile(self, q: float) -> np.ndarray:
            idx = int(np.round((self.num_samples - 1) * float(q)))
            return self._sorted[idx, :]

    rng = np.random.default_rng(0)
    samples = rng.normal(90.0, 1.0, size=(100, 64))     # (S=100, H=64)
    fc = FakeSampleForecast(samples)
    lv = [0.1, 0.5, 0.9]

    full = quantiles_from_forecast(fc, lv, 64)
    check("满 chunk 形状 (Q, H)", full.shape == (3, 64), str(full.shape))
    # 核心不变式：q50 必须**逐位等于**点预测链路用的 fc.quantile(0.5)
    check("q50 逐位等于 fc.quantile(0.5)（点预测链路）",
          bool(np.array_equal(full[1], fc.quantile(0.5))))
    # 并证明它**不等于** np.quantile —— 这正是曾经的分叉来源
    check("q50 不等于 np.quantile(0.5)（取法不同，证明判据有效）",
          not close(float(full[1, 5]), float(np.quantile(samples[:, 5], 0.5))),
          f"{full[1, 5]} vs {np.quantile(samples[:, 5], 0.5)}")

    # 小 chunk：切片发生在 fc.quantile 之后，与 predict_batch 同构
    part = quantiles_from_forecast(fc, lv, 21)
    check("小 chunk 形状 (Q, 21)", part.shape == (3, 21), str(part.shape))
    check("小 chunk q50 逐位等于 fc.quantile(0.5)[:21]",
          bool(np.array_equal(part[1], fc.quantile(0.5)[:21])))
    check("小 chunk 相邻列不同（确实切了时间轴）",
          not close(float(part[1, 3]), float(part[1, 4])))
    # 旧写法 samples[:horizon] 会切样本轴：前 10 条样本算出的中位数必然不同
    few = FakeSampleForecast(rng.normal(90.0, 1.0, size=(10, 64)))
    q10 = quantiles_from_forecast(few, lv, 10)
    check("样本数 < horizon 也正确（旧写法会广播失败）",
          q10.shape == (3, 10), str(q10.shape))
    check("样本少时 q50 仍逐位等于 fc.quantile(0.5)[:10]",
          bool(np.array_equal(q10[1], few.quantile(0.5)[:10])))

    # 层值顺序不能被打乱：q10 <= q50 <= q90（本构造下样本有序统计量保证）
    check("三层单调", bool(np.all(full[0] <= full[1]) and np.all(full[1] <= full[2])))


# ------------------------------------------- 14. CRPS 在 y=q50 处取极小
def test_crps_minimized_at_median() -> None:
    """固定预测分布，CRPS 作为观测 y 的函数**在中位数处达到全局极小**。

    ⚠️ 这条要替代一个曾经写错的直觉：CRPS ≥ |y − q50| 并**不**普遍成立。
    反例（本组第二个断言就是它）：分位数函数近似 U[0,10]（中位数 5），取 y = −100，
        点质量分布 CRPS = |−100 − 5| = 105，
        而该分布的 CRPS ≈ 103.8 < 105。
    也就是说"CRPS 小于同口径 MAE"本身不是 bug——真正的不变式是
    d/dy CRPS = 2F(y) − 1，零点在 y = 中位数，且二阶导 2f(y) > 0。

    ⚠️ 注意"极小"是**一段平台**而不是一个点：离散分位函数下，
    只要 F(y) = 0.5（即 y 落在 [q50, 下一个断点) 区间内），导数就恒为 0。
    所以只能断言"中位数处达到最小值"，不能断言 argmin 唯一。

    判断概率指标是否可信，不能拿它和 MAE 比大小，只能查**两条链路是否同一条
    轨迹**（见第 15 组）。这一组锁住的是 CRPS 定义本身没被写坏。
    """
    print("\n[14] 固定分布下 CRPS 在（诱导分布的）中位数处取全局极小")
    # 9 个等距分位点，q50 = 5
    lv = [0.1 * i for i in range(1, 10)]
    q = np.array([float(i) for i in range(1, 10)])
    q50 = float(q[4])

    # 离散分位函数诱导的分布。注意本实现的约定是**段内取右端点分位**：
    # 段 [α_{i−1}, α_i] 的质量归到 q_i，因此 F 的跳点从 q_0 挪到 q_1 一侧。
    alpha = [0.0] + lv + [1.0]
    qs = [q[0]] + list(q) + [q[-1]]
    cdf = np.cumsum(np.diff(alpha))                          # 长度 = len(alpha)−1
    m_star = float(qs[1 + int(np.argmax(cdf >= 0.5 - 1e-12))])  # 诱导分布的中位数

    grid = np.linspace(-10.0, 20.0, 601)
    vals = np.array([crps_from_quantiles(float(y), q, lv) for y in grid])
    # 断点必须进网格，否则最小值会被"跳过"
    at_q = np.array([crps_from_quantiles(float(v), q, lv) for v in q])
    gmin = min(float(vals.min()), float(at_q.min()))

    check(f"诱导中位数 m*={m_star:g} 与 q50={q50:g} 相差不超过 1 个分位步长",
          abs(m_star - q50) <= 1.0 + 1e-9, f"{m_star} vs {q50}")
    check(f"CRPS 在 m* 处取到全局最小（{crps_from_quantiles(m_star, q, lv):.6f}）",
          abs(crps_from_quantiles(m_star, q, lv) - gmin) < 1e-12,
          f"{crps_from_quantiles(m_star, q, lv):.6f} vs min={gmin:.6f}")
    check("CRPS 处处非负", bool(np.all(vals >= -1e-12)), f"min={vals.min():.3e}")
    # 远离中位数必须严格变大（排除"定义写反/符号错"这类故障）
    at_med = crps_from_quantiles(m_star, q, lv)
    check("y=m*−10 处严格大于 m* 处",
          crps_from_quantiles(m_star - 10.0, q, lv) > at_med + 1e-6)
    check("y=m*+10 处严格大于 m* 处",
          crps_from_quantiles(m_star + 10.0, q, lv) > at_med + 1e-6)

    # 关键反例：支撑外 y 的 CRPS 可以小于 |y − q50|，说明该下界不成立
    y_far = -100.0
    c_far = crps_from_quantiles(y_far, q, lv)
    check("支撑外 y：CRPS < |y−q50|（下界不成立的反例）",
          c_far < abs(y_far - q50) - 1e-6,
          f"CRPS={c_far:.4f} vs |y−q50|={abs(y_far - q50):.4f}")

    # 3 层退化情形：最小值是一段平台，[q50, q90] 内取值全等
    q3, lv3 = np.array([86.0, 90.0, 94.0]), [0.1, 0.5, 0.9]
    plate = [crps_from_quantiles(y, q3, lv3) for y in (90.0, 91.5, 93.9)]
    check("3 层时 [q50,q90] 内 CRPS 恒定（极小是平台不是点）",
          max(plate) - min(plate) < 1e-12, str(plate))
    check("3 层时平台值等于最小值",
          min(plate) <= crps_from_quantiles(85.0, q3, lv3) - 1e-9)


# ------------------------------------- 15. 中位数层与点预测的口径一致性闸门
def test_median_deviation_gate() -> None:
    """`median_deviation()` 必须能抓住"分位数链路与点预测链路分叉"。

    timesfm 2.5 的真实事故：分位数张量是 `len(quantiles)+1 = 10` 列，
    第 0 列是空槽、9 个分位数在第 1..9 列、第 5 列兼作点预测。
    适配器用 `linspace(0.1, 0.9, 10)` 硬铺层值，整体错位一格，
    CALCE 上两条轨迹最大分叉 10.93 SOH%——但当时没有任何告警，
    错的 CRPS/PICP/MPIW 照样写进了 per_cell_*.csv。
    """
    print("\n[15] 中位数层与点预测一致性闸门")
    from evaluation.metrics import median_deviation

    lv = [0.1, 0.5, 0.9]
    p1 = np.array([90.0, 89.5, 89.0])
    q1 = np.vstack([p1 - 1.0, p1, p1 + 1.0])          # q50 逐位等于点预测
    d = median_deviation([p1], [q1], lv)
    check("一致时 max_abs == 0", d["max_abs"] == 0.0, str(d))
    check("一致时 n_bad == 0", d["n_bad"] == 0, str(d))
    check("统计到的 cell 数 = 1", d["n_checked"] == 1, str(d))
    check("median_level == 0.5", d["median_level"] == 0.5, str(d))

    q2 = q1.copy()
    q2[1, 2] += 0.01                                   # 只动一个点
    d2 = median_deviation([p1], [q2], lv)
    check("差 0.01 也判为不一致（严格容差 1e-9）", d2["n_bad"] == 1, str(d2))
    check("偏差量级正确", close(d2["max_abs"], 0.01, 1e-9), str(d2["max_abs"]))

    # 只比 horizon 重叠部分；None 与缺层都要跳过而不是崩
    d3 = median_deviation([None, p1], [q1, None], lv)
    check("None 条目跳过", d3["n_checked"] == 0, str(d3))
    d4 = median_deviation([p1], [np.vstack([p1 - 1.0, p1])], lv)
    check("分位数层数不足时跳过（不崩）", d4["n_checked"] == 0, str(d4))
    d5 = median_deviation([p1[:2]], [q1[:, :3]], lv)
    check("horizon 不等时按重叠长度比对", d5["max_abs"] == 0.0, str(d5))

    # 层值里没有精确 0.5 时，取最接近的层，并如实回报
    d6 = median_deviation([p1], [np.vstack([p1 - 1.0, p1, p1 + 1.0])],
                          [0.1, 0.4, 0.9])
    check("无 0.5 层时退回最接近层 (0.4)", d6["median_level"] == 0.4, str(d6))


# ------------------- 16. 采样模型的随机状态：两条链路必须从同一状态出发
def test_sampling_rng_checkpoint_restore() -> None:
    """计数器推种子的采样模型，两条链路必须**从同一随机状态出发**。

    真实事故（moirai / lag-llama）：适配器用全局单调计数器推种子
        `_calls += 1; seed = base + _calls`
    而点预测链路与分位数链路是**两遍独立遍历**：
        第 1 遍用掉计数器 1..N，第 2 遍接着用 N+1..2N。
    同一个 chunk 在两遍里拿到不同种子、不同样本 ⇒ `q50 ≠ 点预测`。
    实测未修复时 max|p−q50|：moirai CALCE 0.75 / MIT 4.52 / NASA 5.76 SOH%，
    lag-llama CALCE 62.5 / MIT 78.2 / NASA 45.2 SOH%。

    修法是 `rng_checkpoint()` / `rng_restore()`：分位数链路前把计数器复位。
    本组用假模型锁住这个机制，并确认**点预测链路不受影响**。
    """
    print("\n[16] 采样模型的随机状态 checkpoint / restore")
    from adapters.base import Forecaster, rolling_forecast, rolling_forecast_quantiles

    class FakeSampler(Forecaster):
        """用计数器推种子的采样模型，故意复现真实适配器的结构。"""
        supports_batch = True
        max_context = 64
        max_horizon = 16

        def __init__(self) -> None:
            self.name = "fake-sampler"
            self.seed = 1234
            self._calls = 0
            self._rng = np.random.default_rng(0)

        def _seed_next(self) -> None:
            self._calls += 1
            self._rng = np.random.default_rng(int(self.seed) + self._calls)

        def rng_checkpoint(self):
            return int(self._calls)

        def rng_restore(self, ckpt) -> None:
            if ckpt is not None:
                self._calls = int(ckpt)

        def _draw(self, contexts, horizon):
            self._seed_next()
            # 采样：中位数 = 上下文的线性外推 + 一个随种子变化的扰动
            out = []
            for c in contexts:
                base = np.linspace(c[-1], c[-1] - 0.5, horizon)
                out.append(base + self._rng.normal(0, 0.3, size=horizon))
            return np.asarray(out, dtype=float)

        def predict(self, context, horizon):
            return self.predict_batch([context], horizon)[0]

        def predict_batch(self, contexts, horizon):
            return list(self._draw(contexts, horizon))

        def predict_quantiles_batch(self, contexts, horizon, levels=(0.1, 0.5, 0.9)):
            m = self._draw(contexts, horizon)          # (B, H)
            # 约定返回 (B, Q, H)
            return np.stack([m - 1.0, m, m + 1.0], axis=1)

    ctx = 90.0 - 0.02 * np.arange(64)
    H, CH = 48, 16

    # (a) 不恢复状态：两遍遍历种子错开 ⇒ 分叉
    m_bad = FakeSampler()
    p_bad = rolling_forecast(m_bad, ctx, H, CH)
    q_bad = rolling_forecast_quantiles(m_bad, ctx, H, CH, (0.1, 0.5, 0.9))
    dev_bad = float(np.abs(p_bad - q_bad[1]).max())
    check("不恢复随机状态时两条链路分叉（复现事故）", dev_bad > 1e-6, f"{dev_bad:.3e}")

    # (b) 恢复状态：两条链路逐位相同
    m_ok = FakeSampler()
    ckpt = m_ok.rng_checkpoint()
    p_ok = rolling_forecast(m_ok, ctx, H, CH)
    m_ok.rng_restore(ckpt)
    q_ok = rolling_forecast_quantiles(m_ok, ctx, H, CH, (0.1, 0.5, 0.9))
    dev_ok = float(np.abs(p_ok - q_ok[1]).max())
    check("恢复随机状态后 q50 与点预测逐位相同", dev_ok == 0.0, f"{dev_ok:.3e}")

    # (c) 关键：恢复不改变点预测链路的结果
    check("恢复状态不影响点预测链路（主表数字不变）",
          bool(np.array_equal(p_ok, p_bad)))

    # (d) 分位数链路跑完后计数器回到原值，后续分组不受影响
    m_after = FakeSampler()
    c1 = m_after.rng_checkpoint()
    rolling_forecast(m_after, ctx, H, CH)
    end_point = m_after.rng_checkpoint()
    m_after.rng_restore(c1)
    rolling_forecast_quantiles(m_after, ctx, H, CH, (0.1, 0.5, 0.9))
    check("分位数链路结束后计数器与原序列一致", m_after.rng_checkpoint() == end_point,
          f"{m_after.rng_checkpoint()} vs {end_point}")

    # (e) 确定性模型（默认实现）不受影响
    class Det(Forecaster):
        name = "det"
        def predict(self, context, horizon):
            return np.full(horizon, float(context[-1]))

    d = Det()
    check("确定性模型 rng_checkpoint 返回 None", d.rng_checkpoint() is None)
    d.rng_restore(None)   # 不应抛错
    check("确定性模型 rng_restore(None) 安全", True)
    dp = rolling_forecast(d, ctx, H, CH)
    check("确定性模型点预测正常", dp.shape == (H,), str(dp.shape))
    try:
        rolling_forecast_quantiles(d, ctx, H, CH, (0.1, 0.5, 0.9))
        check("确定性模型不支持下分位数时应抛错", False, "没有抛错")
    except NotImplementedError:
        check("确定性模型不支持下分位数时应抛错", True)


if __name__ == "__main__":
    print("=" * 64)
    print("概率指标 + EOL 重设计 回归测试")
    print("=" * 64)
    test_crps_degenerate_equals_abs_error()
    test_crps_three_levels_degenerate()
    test_crps_matches_exact_sampling()
    test_crps_nonneg_and_min_at_median()
    test_picp_mpiw()
    test_custom_interval()
    test_eol_redesign()
    test_eol_gate_still_works()
    test_rolling_quantiles_feeds_median_only()
    test_unsupported_model_raises()
    test_baselines_report_no_quantiles()
    test_repair_quantiles_pins_median()
    test_quantiles_from_forecast_matches_point()
    test_crps_minimized_at_median()
    test_median_deviation_gate()
    test_sampling_rng_checkpoint_restore()
    print("\n" + "=" * 64)
    print(f"通过 {len(_PASS)} 项，失败 {len(_FAIL)} 项")
    if _FAIL:
        for f in _FAIL:
            print(f"  ✗ {f}")
        sys.exit(1)
    print("全部通过 ✓")
