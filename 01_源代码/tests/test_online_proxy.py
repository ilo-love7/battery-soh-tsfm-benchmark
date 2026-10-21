"""在线代理（§9.5）回归测试。

无第三方依赖（不需要 pytest），直接 `python tests/test_online_proxy.py` 跑。

锁死四类容易悄悄坏掉的性质：

1. **`slope_*` 必须按 horizon 归一化**。三个库的 horizon 差 10 倍（NASA 84 /
   CALCE ~350 / MIT 924），不除就会把"库识别"当成"衰减速率"，
   整个 §9.5 的结论会变成"特征只是在认库"。这里用同一条曲线、两个 horizon
   来卡：数值必须严格按 horizon 线性放大。
2. **`r2_lin` / `rel_noise` 的基准**。线性序列必须给出 `r2_lin=1`、`rel_noise=0`，
   否则后面"噪声大 → 该信线性外推"这条判据的方向就无从谈起。
3. **`spearman` 在退化输入上必须返回 NaN 而不是 0**。常数序列、样本不足时返回 0
   会让"最强特征"的排序静默选错（0 会排在真负相关前面）。
4. **`w_cov_cell` 的删失语义**。扫描到上界仍未达标时必须返回上界 + `censored=True`，
   不能被当成"这个 w 就够用了"——§10.8 与 §9.5 的很多结论都建立在这个标记上。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from online_proxy import (ctx_features, picp_at, route_eval, spearman,   # noqa: E402
                          w_cov_cell, best_rule, FEATURES)

_PASS: list = []
_FAIL: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  — {detail}" if detail and not cond else ""))


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# --------------------------------------------------------------- 1. 特征基本量
def test_features_on_linear_series():
    print("\n[1] 线性序列：斜率/曲率/拟合优度必须自洽")
    ctx = 100.0 - 0.05 * np.arange(400)          # 每步 −0.05，共掉 19.95
    f = ctx_features(ctx, horizon=200)
    check("drop_ctx = 上下文总衰减 %", close(f["drop_ctx"], 100 * 19.95 / 100.0, 1e-6),
          f"{f['drop_ctx']}")
    # ⚠️ 归一化基准是**上下文末值**（x[-1]=80.05），不是首值：外推降幅要相对"当前容量"算，
    # 相对首值算会让"已经掉了很多"的序列系统性偏小。
    check("slope_all = 按全段斜率外推 200 步的降幅 %（基准是末值）",
          close(f["slope_all"], 100 * 0.05 * 200 / 80.05, 1e-6), f"{f['slope_all']}")
    check("线性序列 slope_recent ≈ slope_all", close(f["slope_recent"], f["slope_all"], 1e-6),
          f"{f['slope_recent']} vs {f['slope_all']}")
    check("线性序列 accel_ctx ≈ 1", close(f["accel_ctx"], 1.0, 1e-6), f"{f['accel_ctx']}")
    check("线性序列 r2_lin ≈ 1", close(f["r2_lin"], 1.0, 1e-6), f"{f['r2_lin']}")
    check("线性序列 rel_noise ≈ 0", abs(f["rel_noise"]) < 1e-9, f"{f['rel_noise']}")
    check("线性序列 curv_quad ≈ 0", abs(f["curv_quad"]) < 1e-6, f"{f['curv_quad']}")
    check("n_ctx 记录上下文长度", close(f["n_ctx"], 400.0))


# ------------------------------------------------------- 2. horizon 归一化必须成立
def test_slope_scales_with_horizon():
    print("\n[2] horizon 归一化：同一条曲线，horizon 翻倍 ⇒ slope_* 翻倍")
    ctx = 100.0 - 0.05 * np.arange(400)
    a = ctx_features(ctx, horizon=100)
    b = ctx_features(ctx, horizon=200)
    check("slope_all 随 horizon 线性放大", close(b["slope_all"], 2 * a["slope_all"], 1e-9),
          f"{a['slope_all']} -> {b['slope_all']}")
    check("slope_recent 随 horizon 线性放大",
          close(b["slope_recent"], 2 * a["slope_recent"], 1e-9),
          f"{a['slope_recent']} -> {b['slope_recent']}")
    check("drop_ctx 与 horizon 无关（累计量不是速率）",
          close(a["drop_ctx"], b["drop_ctx"], 1e-12))
    check("r2_lin 与 horizon 无关", close(a["r2_lin"], b["r2_lin"], 1e-12))


# ----------------------------------------------------------- 3. 加速衰减的符号
def test_accelerating_series():
    print("\n[3] 加速衰减：accel_ctx > 1、curv_quad > 0")
    t = np.arange(400, dtype=float)
    ctx = 100.0 - 0.0002 * t ** 2            # 越来越快
    f = ctx_features(ctx, horizon=200)
    check("accel_ctx > 1（后段更快）", f["accel_ctx"] > 1.0, f"{f['accel_ctx']}")
    check("curv_quad > 0（二次外推比线性多掉）", f["curv_quad"] > 0, f"{f['curv_quad']}")
    check("slope_recent > slope_all", f["slope_recent"] > f["slope_all"],
          f"{f['slope_recent']} vs {f['slope_all']}")


def test_short_context_returns_empty():
    print("\n[4] 上下文过短 / horizon 非法时返回空 dict（不崩、不静默给 0）")
    check("n=7 返回 {}", ctx_features(np.linspace(100, 99, 7), 10) == {})
    check("horizon=0 返回 {}", ctx_features(np.linspace(100, 99, 50), 0) == {})


# ------------------------------------------------------------------ 5. Spearman
def test_spearman():
    print("\n[5] Spearman：单调、反向、退化输入")
    x = np.arange(20, dtype=float)
    check("严格单调 ⇒ 1", close(spearman(x, x ** 3), 1.0, 1e-12))
    check("严格反向 ⇒ −1", close(spearman(x, -x), -1.0, 1e-12))
    check("秩相关不受单调变换影响", close(spearman(x, np.exp(x / 10)), 1.0, 1e-12))
    check("常数序列 ⇒ NaN（不能返回 0）", np.isnan(spearman(x, np.ones_like(x))))
    check("样本不足 ⇒ NaN", np.isnan(spearman(np.array([1.0, 2.0]), np.array([2.0, 1.0]))))
    check("含 NaN 的行被剔除后仍能算",
          close(spearman(np.array([1.0, 2, 3, 4, np.nan]),
                         np.array([1.0, 2, 3, 4, 5])), 1.0, 1e-12))


# ------------------------------------------------------------- 6. PICP 与 w_cov
def _quantiles(levels, horizon, half):
    """构造一个"半宽 = half"的对称分位张量，形状 (Q, H)。"""
    lv = np.asarray(levels, dtype=float)
    mid = 90.0
    q = mid + (lv[:, None] - 0.5) * 2 * half
    return np.repeat(q, horizon, axis=1)


def test_picp_and_wcov():
    print("\n[6] PICP 与 w_cov：单调性 + 删失语义")
    levels = [0.1, 0.5, 0.9]
    truth = np.full(100, 90.0)
    q = _quantiles(levels, 100, 5.0)
    check("真值在区间内 ⇒ PICP=1", close(picp_at(q, truth, 1.0, 1), 1.0))
    truth_out = np.full(100, 120.0)                     # 全在区间外
    check("真值在区间外 ⇒ PICP=0", close(picp_at(q, truth_out, 1.0, 1), 0.0))
    check("w 变大 PICP 不减", picp_at(q, truth_out, 100.0, 1)
          >= picp_at(q, truth_out, 10.0, 1))

    ws = np.array([0.1, 0.5, 1.0, 10.0, 100.0])
    w, cen = w_cov_cell(q, truth, levels, ws)
    check("已达标时返回最小网格点", close(w, 0.1) and not cen, f"{w}, censored={cen}")
    w2, cen2 = w_cov_cell(q, np.full(100, 1e6), levels, ws)
    check("永不达标时返回上界并标 censored",
          close(w2, 100.0) and cen2, f"{w2}, censored={cen2}")


# ------------------------------------------------------------------- 7. 路由
def test_route_eval():
    print("\n[7] route_eval：方向、阈值、MAE 计算")
    df = pd.DataFrame({
        "f": [1.0, 2.0, 3.0, 4.0],
        "mae_chronos-bolt": [1.0, 1.0, 9.0, 9.0],
        "mae_naive_linear": [9.0, 9.0, 1.0, 1.0],
    })
    mae, pick = route_eval(df, "f", 2.5, "chronos-bolt", higher_means_fm=False)
    check("小 f 选 FM、大 f 选线性 ⇒ MAE=1", close(mae, 1.0), f"{mae}")
    check("选 FM 的比例 = 0.5", close(pick, 0.5), f"{pick}")
    mae2, pick2 = route_eval(df, "f", 2.5, "chronos-bolt", higher_means_fm=True)
    check("方向反过来 ⇒ 全选错，MAE=9", close(mae2, 9.0), f"{mae2}")
    check("全选错时 pick_fm = 0.5", close(pick2, 0.5), f"{pick2}")


def test_best_rule_needs_enough_cells():
    print("\n[8] best_rule：样本 < 10 时放弃（NASA 只有 4 支）")
    df = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0],
                       "mae_chronos-bolt": [1.0, 1.0, 9.0, 9.0],
                       "mae_naive_linear": [9.0, 9.0, 1.0, 1.0]})
    r = best_rule(df, "chronos-bolt")
    check("n=4 返回 feature=None", r["feature"] is None, str(r))
    check("n=4 时 MAE 为 NaN（不是 inf）", np.isnan(r["mae"]), str(r["mae"]))


def test_feature_list_matches_output():
    print("\n[9] FEATURES 与 ctx_features 的输出必须一一对应")
    f = ctx_features(100.0 - 0.05 * np.arange(400), horizon=200)
    check("FEATURES 中每个名字都在输出里", all(k in f for k in FEATURES),
          str([k for k in FEATURES if k not in f]))
    check("输出里没有 FEATURES 之外的量（horizon 除外）",
          set(f) - set(FEATURES) == {"horizon"}, str(set(f) - set(FEATURES)))


if __name__ == "__main__":
    print("=" * 64)
    print("在线代理（§9.5）回归测试")
    print("=" * 64)
    test_features_on_linear_series()
    test_slope_scales_with_horizon()
    test_accelerating_series()
    test_short_context_returns_empty()
    test_spearman()
    test_picp_and_wcov()
    test_route_eval()
    test_best_rule_needs_enough_cells()
    test_feature_list_matches_output()
    print("\n" + "=" * 64)
    print(f"通过 {len(_PASS)} 项，失败 {len(_FAIL)} 项")
    if _FAIL:
        for f in _FAIL:
            print(f"  ✗ {f}")
        sys.exit(1)
    print("全部通过 ✓")
