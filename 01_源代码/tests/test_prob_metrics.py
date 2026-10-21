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


# ------------------------------------------------------- 13. 采样型分位数切轴
def test_quantiles_from_samples_slices_time_axis() -> None:
    """`samples[:, :horizon]`，不是 `samples[:horizon]`。

    Moirai / Lag-Llama 真实踩过的坑：predictor 固定出 64 步，
    rolling 到最后一个小 chunk（horizon=21）时
      - 写成 samples[:horizon] 切的是**样本轴** → 广播失败
        （实测 could not broadcast (3,64) into (3,21)）；
      - 且 horizon < num_samples 时即使不崩，也只用了前 horizon 条样本，
        中位数与 predict_batch 的 fc.quantile(0.5)（全部样本）不一致。
    """
    print("\n[13] 采样张量切轴：时间轴是最后一维")
    from adapters.base import quantiles_from_samples

    rng = np.random.default_rng(0)
    samples = rng.normal(90.0, 1.0, size=(100, 64))     # (S=100, H=64)
    lv = [0.1, 0.5, 0.9]

    full = quantiles_from_samples(samples, lv, 64)
    check("满 chunk 形状 (Q, H)", full.shape == (3, 64), str(full.shape))
    check("满 chunk 中位数 == 该列全样本分位",
          close(float(full[1, 5]), float(np.quantile(samples[:, 5], 0.5))))
    # 关键不变量：中位数必须用**全部**样本，才与 fc.quantile(0.5) 同源。
    # "前 64 条"与"全部 100 条"在同一列上必然不同（随机序列），可作判据。
    check("中位数用全部样本（而非前 horizon 条）",
          not close(float(full[1, 5]), float(np.quantile(samples[:64, 5], 0.5))),
          f"{full[1, 5]} vs {np.quantile(samples[:64, 5], 0.5)}")

    part = quantiles_from_samples(samples, lv, 21)
    check("小 chunk 形状 (Q, 21)", part.shape == (3, 21), str(part.shape))
    check("小 chunk 中位数 == 同列全样本分位",
          close(float(part[1, 7]), float(np.quantile(samples[:, 7], 0.5))))
    check("小 chunk 相邻列不同（确实切了时间轴）",
          not close(float(part[1, 3]), float(part[1, 4])))

    # 旧写法的失败模式：horizon >= num_samples 时 samples[:horizon] 形状不变但语义错
    few = rng.normal(90.0, 1.0, size=(10, 64))
    q10 = quantiles_from_samples(few, lv, 10)
    check("样本数 < horizon 也正确（旧写法会广播失败）",
          q10.shape == (3, 10), str(q10.shape))
    check("样本少时中位数仍用全部 10 条",
          close(float(q10[1, 3]), float(np.quantile(few[:, 3], 0.5))))

    # 时间轴不足 / 维度不对必须明确报错，不能静默返回短序列
    try:
        quantiles_from_samples(samples, lv, 100)
        check("时间轴不足应报错", False, "没有抛错")
    except ValueError:
        check("时间轴不足应报错", True)
    try:
        quantiles_from_samples(np.ones(5), lv, 3)
        check("一维输入应报错", False, "没有抛错")
    except ValueError:
        check("一维输入应报错", True)


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
    test_quantiles_from_samples_slices_time_axis()
    print("\n" + "=" * 64)
    print(f"通过 {len(_PASS)} 项，失败 {len(_FAIL)} 项")
    if _FAIL:
        for f in _FAIL:
            print(f"  ✗ {f}")
        sys.exit(1)
    print("全部通过 ✓")
