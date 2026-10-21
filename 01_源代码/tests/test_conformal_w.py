"""conformal_w.py 的回归测试：伪任务切分 + 前提 1 统计。

只测纯逻辑，不跑模型推理（`run_one` 需要权重，属于集成测试）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from conformal_w import T_MIN, build_pseudo, premise1_stats  # noqa: E402

N = 0
FAIL = []


def check(name: str, cond: bool, extra: str = "") -> None:
    global N
    N += 1
    if not cond:
        FAIL.append(f"{name}{(' — ' + extra) if extra else ''}")
        print(f"  ✗ {name}{(' — ' + extra) if extra else ''}")


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def test_split_is_exact_partition():
    """伪上下文 + 伪真值必须正好等于原上下文，一点不重叠、一点不遗漏。"""
    ctx = np.arange(300, dtype=float)
    got = build_pseudo(ctx, 0.5)
    check("build_pseudo 不返回 None", got is not None)
    a, b, hp = got
    check("拼接后与原文逐位相等", np.array_equal(np.concatenate([a, b]), ctx))
    check("Hp = round(T*hfrac)", hp == int(round(300 * 0.5)), f"hp={hp}")
    check("伪上下文长度 = T - Hp", a.size == 300 - hp, f"{a.size}")
    check("伪真值长度 = Hp", b.size == hp, f"{b.size}")
    check("伪真值是尾段", b[0] == ctx[300 - hp] and b[-1] == ctx[-1])


def test_t_min_floor():
    """序列太短时 Hp 被 T-T_MIN 夹住，保证伪上下文至少 T_MIN 点。"""
    ctx = np.arange(100, dtype=float)
    a, b, hp = build_pseudo(ctx, 0.5)
    check("T=100 时 Hp 被夹到 T-T_MIN", hp == 100 - T_MIN, f"hp={hp}")
    check("伪上下文 = T_MIN", a.size == T_MIN, f"{a.size}")


def test_too_short_returns_none():
    """T 太小（Hp < 16）必须返回 None，不能返回一段垃圾伪任务。"""
    check("T=70 返回 None", build_pseudo(np.arange(70, dtype=float), 0.5) is None)
    check("T=79 返回 None", build_pseudo(np.arange(79, dtype=float), 0.5) is None)
    check("T=80 不返回 None", build_pseudo(np.arange(80, dtype=float), 0.5) is not None)


def test_hfrac_controls_horizon():
    """hfrac 变小 ⇒ 伪 horizon 变短、伪上下文变长。"""
    ctx = np.arange(400, dtype=float)
    a25, b25, h25 = build_pseudo(ctx, 0.25)
    a50, b50, h50 = build_pseudo(ctx, 0.50)
    check("hfrac 0.25 的 Hp 是 0.5 的一半", h25 * 2 == h50, f"{h25} vs {h50}")
    check("hfrac 越小伪上下文越长", a25.size > a50.size)


def test_premise1_stats():
    """构造已知比值的 inv 表，验证中位比值与配对逻辑。"""
    rows = []
    # 20 个 cell：w(50%) = 0.1 × w(100%)，w(25%) = 0.02 × w(100%)
    for i in range(20):
        w100 = 100.0 * (1 + i)
        for frac, w in ((0.25, 0.02 * w100), (0.5, 0.10 * w100), (1.0, w100)):
            rows.append({"dataset": "d", "model": "m", "cell": f"c{i}",
                         "frac_of_horizon": frac, "w_cov": w, "censored": False})
    st = premise1_stats(pd.DataFrame(rows))
    check("统计里有 0.5 档", 0.5 in st)
    check("0.5 档中位比值 = 0.10", close(st[0.5][1], 0.10, 1e-12), f"{st[0.5][1]}")
    check("0.25 档中位比值 = 0.02", close(st[0.25][1], 0.02, 1e-12), f"{st[0.25][1]}")
    check("IQR 退化成一个点", close(st[0.5][2], 0.10, 1e-12) and close(st[0.5][3], 0.10, 1e-12))
    check("log ρ = 1（严格幂律）", close(st[0.5][0], 1.0, 1e-9), f"{st[0.5][0]}")


def test_premise1_needs_enough_pairs():
    """配对太少（<4）时不能报出统计，否则会给出噪声比值。"""
    rows = []
    for i in range(3):
        w100 = 100.0 * (1 + i)
        for frac, w in ((0.5, 0.1 * w100), (1.0, w100)):
            rows.append({"dataset": "d", "model": "m", "cell": f"c{i}",
                         "frac_of_horizon": frac, "w_cov": w, "censored": False})
    st = premise1_stats(pd.DataFrame(rows))
    check("n=3 时不报 0.5 档", 0.5 not in st, f"keys={list(st)}")


def test_premise1_ignores_nonpositive():
    """w_cov = 0（删失到 0）会让 log 炸掉，必须被剔除。"""
    rows = []
    for i in range(6):
        w100 = 100.0 * (1 + i)
        for frac, w in ((0.5, 0.1 * w100), (1.0, w100)):
            rows.append({"dataset": "d", "model": "m", "cell": f"c{i}",
                         "frac_of_horizon": frac, "w_cov": w, "censored": False})
    rows.append({"dataset": "d", "model": "m", "cell": "bad",
                 "frac_of_horizon": 0.5, "w_cov": 0.0, "censored": True})
    rows.append({"dataset": "d", "model": "m", "cell": "bad",
                 "frac_of_horizon": 1.0, "w_cov": 500.0, "censored": False})
    st = premise1_stats(pd.DataFrame(rows))
    check("剔除 w=0 后中位比值仍为 0.1", close(st[0.5][1], 0.1, 1e-12), f"{st[0.5][1]}")


def main() -> int:
    tests = [test_split_is_exact_partition, test_t_min_floor, test_too_short_returns_none,
             test_hfrac_controls_horizon, test_premise1_stats,
             test_premise1_needs_enough_pairs, test_premise1_ignores_nonpositive]
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print(f"\n{len(tests)} 组 / {N} 项断言，失败 {len(FAIL)}")
    if FAIL:
        for x in FAIL:
            print("  !", x)
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
