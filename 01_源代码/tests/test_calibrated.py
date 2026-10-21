"""校准头（§9.3）回归测试。

无第三方依赖（不需要 pytest），直接 `python tests/test_calibrated.py` 跑。

锁死四类容易悄悄坏掉的性质：
1. 恒等校准 (s=1, b=0) 必须与冻结基础模型**逐位一致**——否则"校准版"的
   点预测链路与已发表版不可比，所有对比数字作废；
2. s ≤ 0 必须拒绝（分位数序会反转，单调性是这一族校准的前提）；
3. 分位数链路与点预测链路必须自洽（同一批参数、同一单调变换），否则
   CRPS/PICP 与 MAE 说的不是同一个模型；
4. 两个参数的**递归语义**（b = 每步漂移、s = 每步变化量缩放）必须成立，
   否则论文里对参数的解读就是错的。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adapters import rolling_forecast                              # noqa: E402
from adapters.base import Forecaster                               # noqa: E402
from adapters.calibrated import CalibratedForecaster, calibrate    # noqa: E402

_PASS: list = []
_FAIL: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  — {detail}" if detail and not cond else ""))


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def arr_close(a, b, tol: float = 1e-9) -> bool:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=tol, atol=tol))


class FakeConst(Forecaster):
    """恒等预测器：预测"未来 = 上下文最后一个点"。用来隔离校准头本身的行为。"""
    name = "fake-const"

    def predict(self, context, horizon):
        return np.full(horizon, float(context[-1]))

    def supports_quantiles(self):
        return True

    def predict_quantiles_batch(self, contexts, horizon, levels=(0.1, 0.5, 0.9)):
        lv = list(levels)
        out = np.empty((len(contexts), len(lv), horizon))
        for i, c in enumerate(contexts):
            a = float(c[-1])
            for j, l in enumerate(lv):
                # 低分位数在锚下方、高分位数在锚上方（q10 < q50 < q90）
                z = -1.0 if l < 0.5 else (0.0 if l == 0.5 else 1.0)
                out[i, j, :] = a + z * 0.5      # 以锚为中心的 ±0.5 区间
        return out


class FakeSlope(Forecaster):
    """预测"每步固定跌 d"。用来验证 s 的语义。"""
    name = "fake-slope"

    def __init__(self, d=0.05):
        self.d = d

    def predict(self, context, horizon):
        c = float(context[-1])
        return c - self.d * np.arange(1, horizon + 1)


class PointOnly(Forecaster):
    name = "point-only"

    def predict(self, context, horizon):
        return np.full(horizon, float(context[-1]))


# ---------------------------------------------------------------- 1. 恒等
def test_identity_is_bitwise_equal_to_base() -> None:
    """(s=1, b=0) 必须与冻结基础模型逐位一致——点/批/递归三个口径都要查。"""
    print("\n[1] 恒等校准 == 冻结基础模型（逐位）")
    base, cal = FakeConst(), CalibratedForecaster(FakeConst(), s=1.0, b=0.0)
    ctx = np.linspace(100, 95, 20)
    check("单条逐位一致", np.array_equal(base.predict(ctx, 7), cal.predict(ctx, 7)))

    ctxs = [np.linspace(100, 95, 20), np.linspace(90, 80, 15)]
    for x, y in zip(base.predict_batch(ctxs, 5), cal.predict_batch(ctxs, 5)):
        check("批量逐位一致", np.array_equal(x, y))

    ra = rolling_forecast(base, ctx, 30, chunk=8)
    rb = rolling_forecast(cal, ctx, 30, chunk=8)
    check("递归逐位一致（最关键）", np.array_equal(ra, rb))


def test_calibrate_helper_broadcasting() -> None:
    print("\n[2] calibrate() 广播语义")
    out = calibrate(np.array([[100.0, 99.0, 98.0]]), np.array([[100.0]]), 2.0, 1.0)
    check("c + s(p-c) + b", arr_close(out, [[101.0, 99.0, 97.0]]), str(out))


# ---------------------------------------------------------------- 3. 单调性
def test_nonpositive_slope_rejected() -> None:
    print("\n[3] s ≤ 0 必须拒绝（分位数序会反转）")
    for bad in (0.0, -1.0, -0.5):
        try:
            CalibratedForecaster(FakeConst(), s=bad, b=0.0)
            check(f"s={bad} 被拒绝", False, "没有抛错")
        except ValueError as e:
            check(f"s={bad} 被拒绝", "s 必须 > 0" in str(e), str(e))
    m = CalibratedForecaster(FakeConst(), s=0.01, b=-5.0)
    check("极小正 s 可用", m.s == 0.01 and m.b == -5.0)


# ---------------------------------------------------------------- 4. 参数语义
def test_bias_is_per_step_drift() -> None:
    """基础模型完全不动时，轨迹必须以斜率 b 线性漂移。"""
    print("\n[4] b 的递归语义 = 每步漂移")
    ctx = np.full(10, 100.0)
    out = rolling_forecast(CalibratedForecaster(FakeConst(), s=1.0, b=-0.5),
                           ctx, 6, chunk=2)
    want = [99.5, 99.5, 99.0, 99.0, 98.5, 98.5]
    check("每 chunk 漂移 b", arr_close(out, want), f"{out} vs {want}")

    out2 = rolling_forecast(CalibratedForecaster(FakeConst(), s=1.0, b=0.0),
                            ctx, 6, chunk=2)
    check("b=0 时不动", arr_close(out2, np.full(6, 100.0)))


def test_slope_scales_model_step_change() -> None:
    print("\n[5] s 的语义 = 模型每步变化量的缩放")
    ctx = np.full(10, 100.0)
    base = FakeSlope(d=0.1)
    o_half = CalibratedForecaster(base, s=0.5, b=0.0).predict(ctx, 4)
    o_double = CalibratedForecaster(base, s=2.0, b=0.0).predict(ctx, 4)
    check("s=0.5", arr_close(o_half, [99.95, 99.9, 99.85, 99.8]), str(o_half))
    check("s=2.0", arr_close(o_double, [99.8, 99.6, 99.4, 99.2]), str(o_double))
    o_mix = CalibratedForecaster(base, s=2.0, b=-0.2).predict(ctx, 3)
    check("s 与 b 叠加", arr_close(o_mix, [99.6, 99.4, 99.2]), str(o_mix))


# ---------------------------------------------------------------- 6. 分位数自洽
def test_quantiles_stay_monotone_and_consistent_with_point() -> None:
    print("\n[6] 分位数链路与点预测链路自洽")
    base, cal = FakeConst(), CalibratedForecaster(FakeConst(), s=1.7, b=-0.3)
    ctxs = [np.linspace(100, 95, 20), np.full(12, 90.0)]
    levels = [0.1, 0.5, 0.9]
    q = cal.predict_quantiles_batch(ctxs, 6, levels)
    check("形状 (B,Q,H)", q.shape == (2, 3, 6), str(q.shape))
    check("s>0 保序（q10 ≤ q50 ≤ q90）", bool(np.all(np.diff(q, axis=1) >= -1e-12)))

    p = np.asarray(cal.predict_batch(ctxs, 6))
    check("点预测 == 分位数中位数", arr_close(p, q[:, 1, :]))

    qb = base.predict_quantiles_batch(ctxs, 6, levels)
    ok = True
    for i, c in enumerate([95.0, 90.0]):
        ok = ok and arr_close(q[i], c + 1.7 * (qb[i] - c) - 0.3)
    check("等于 c + s(q_base - c) + b", ok)


def test_quantile_width_shrinks_with_smaller_s() -> None:
    print("\n[7] s 控制区间宽度（修过度自信/发散的方向开关）")
    ctx = np.full(10, 100.0)
    q1 = CalibratedForecaster(FakeConst(), 1.0, 0.0).predict_quantiles(ctx, 5, [0.1, 0.5, 0.9])
    q2 = CalibratedForecaster(FakeConst(), 0.5, 0.0).predict_quantiles(ctx, 5, [0.1, 0.5, 0.9])
    w1 = float(np.mean(q1[2] - q1[0]))
    w2 = float(np.mean(q2[2] - q2[0]))
    check("s=0.5 区间宽度减半", close(w2, 0.5 * w1) and w2 < w1, f"{w1} vs {w2}")


def test_base_without_quantiles_propagates() -> None:
    print("\n[8] 基础模型不支持分位数 ⇒ 明确抛 NotImplementedError")
    cal = CalibratedForecaster(PointOnly(), s=1.0, b=0.0)
    check("supports_quantiles() 透传 False", cal.supports_quantiles() is False)
    try:
        cal.predict_quantiles(np.full(5, 100.0), 3)
        check("抛 NotImplementedError", False, "没有抛错")
    except NotImplementedError:
        check("抛 NotImplementedError", True)


# ---------------------------------------------------------------- 9. 元信息
def test_info_reports_params_and_base() -> None:
    print("\n[9] info() 暴露校准参数（写进 summary.json，可追溯）")
    info = CalibratedForecaster(FakeConst(), s=1.3, b=-0.7).info()
    check("calibrated 标记", info["calibrated"] is True)
    check("cal_params", info["cal_params"] == {"s": 1.3, "b": -0.7}, str(info["cal_params"]))
    check("base_model", info["base_model"] == "fake-const")
    check("公式入档", info["cal_formula"] == "p' = c + s*(p - c) + b")
    check("name 带参数", info["name"] == "fake-const+cal(s=1.3,b=-0.7)", info["name"])


def test_capabilities_passthrough() -> None:
    print("\n[10] 能力透传（不改变被包装模型的限制）")

    class Limited(Forecaster):
        name = "limited"
        max_context = 32
        max_horizon = 8
        supports_batch = True

        def predict(self, context, horizon):
            return np.full(horizon, float(context[-1]))

    cal = CalibratedForecaster(Limited(), s=1.0, b=0.0)
    check("max_context", cal.max_context == 32)
    check("max_horizon", cal.max_horizon == 8)
    check("supports_batch", cal.supports_batch is True)


def test_context_limit_propagates_to_base() -> None:
    """给包装器设上限必须**真的**夹紧被包装的模型。

    run_benchmark 做公平对照时会写 `model.max_context = 512`。
    第一版 CalibratedForecaster 只是在构造时拷贝了一份数值，
    于是包装器显示 512、底层 Chronos 仍是 2048 —— 校准跑用 2 倍长的上下文，
    与零样本基线不同口径，对比直接作废（MIT 上实测差出 0.2 SOH%）。
    """
    print("\n[11] 上下文/步长上限必须下传到 base（否则公平对照失效）")

    class Base(Forecaster):
        name = "b"
        max_context = 2048
        max_horizon = 64
        supports_batch = True

        def predict(self, context, horizon):
            return np.full(horizon, float(context[-1]))

    base = Base()
    cal = CalibratedForecaster(base, s=1.0, b=0.0)
    cal.max_context = 512                       # run_benchmark 就是这么写的
    check("包装器读到 512", cal.max_context == 512)
    check("底层也被夹到 512", base.max_context == 512, f"base={base.max_context}")
    check("_truncate 真的生效", len(base._truncate(np.arange(1000.0))) == 512,
          str(len(base._truncate(np.arange(1000.0)))))

    cal.max_horizon = 16
    check("max_horizon 也下传", base.max_horizon == 16, f"base={base.max_horizon}")


if __name__ == "__main__":
    print("=" * 64)
    print("2 参数单调校准头 回归测试")
    print("=" * 64)
    test_identity_is_bitwise_equal_to_base()
    test_calibrate_helper_broadcasting()
    test_nonpositive_slope_rejected()
    test_bias_is_per_step_drift()
    test_slope_scales_model_step_change()
    test_quantiles_stay_monotone_and_consistent_with_point()
    test_quantile_width_shrinks_with_smaller_s()
    test_base_without_quantiles_propagates()
    test_info_reports_params_and_base()
    test_capabilities_passthrough()
    test_context_limit_propagates_to_base()
    print("\n" + "=" * 64)
    print(f"通过 {len(_PASS)} 项，失败 {len(_FAIL)} 项")
    if _FAIL:
        for f in _FAIL:
            print(f"  ✗ {f}")
        sys.exit(1)
    print("全部通过 ✓")
