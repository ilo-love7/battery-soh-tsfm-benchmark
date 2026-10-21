"""诊断：CALCE 每个 zip 里"真正全深度"的循环有多少。

判定依据（物理口径，不依赖容量计数器）：
    一个循环若为满充满放，必须同时满足
        Vmax ≥ 4.15 V   （充到上限）
        Vmin ≤ 2.85 V   （放到下限）
CALCE 的 CS2/CX2 电芯额定上限 4.2 V、放电截止 2.7 V，留 0.05–0.15 V 容差。

输出每支电池：
    n_all    总循环数（能解析出容量的）
    n_full   全深度循环数
    frac     全深度占比
    ah_full  全深度循环的放电量中位数
    ah_part  非全深度循环的放电量中位数（应显著更小）

用法：python diag_calce_depth.py [--cells CS2_7 CX2_16 ...]
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs.config import DATASETS  # noqa: E402
from data.loaders import _calce_sort_key  # noqa: E402

V_HI = 4.15
V_LO = 2.85


def one_zip(zp: Path) -> dict | None:
    cell = zp.stem
    try:
        zf = zipfile.ZipFile(zp)
    except Exception:
        return None
    names = [n for n in zf.namelist()
             if n.lower().endswith((".xlsx", ".xls"))
             and not n.startswith("__MACOSX")
             and not n.rsplit("/", 1)[-1].startswith("~$")]
    names.sort(key=lambda n: _calce_sort_key(n, cell))

    ahs, full = [], []
    for n in names:
        try:
            sh = pd.read_excel(io.BytesIO(zf.read(n)), sheet_name=None, engine="calamine")
        except Exception:
            continue
        df = next((s for s in sh.values()
                   if s is not None and {"Cycle_Index", "Current(A)", "Test_Time(s)",
                                         "Voltage(V)"} <= set(s.columns)), None)
        if df is None or df.empty:
            continue
        df = df[["Cycle_Index", "Current(A)", "Test_Time(s)", "Voltage(V)"]].apply(
            pd.to_numeric, errors="coerce").dropna()
        if df.empty:
            continue
        df = df.sort_values("Test_Time(s)")
        dt = df["Test_Time(s)"].diff().clip(lower=0, upper=120).fillna(0.0)
        cur = df["Current(A)"]
        ah = (-cur.where(cur < -0.01, 0.0) * dt) / 3600.0
        g = pd.DataFrame({
            "ah": ah.groupby(df["Cycle_Index"]).sum(),
            "vmin": df.groupby("Cycle_Index")["Voltage(V)"].min(),
            "vmax": df.groupby("Cycle_Index")["Voltage(V)"].max(),
        })
        g = g[g.ah > 1e-4]
        if g.empty:
            continue
        is_full = (g.vmin <= V_LO) & (g.vmax >= V_HI)
        ahs.extend(g.ah.tolist())
        full.extend(is_full.tolist())

    if not ahs:
        return None
    ahs = np.asarray(ahs, dtype=float)
    full = np.asarray(full, dtype=bool)
    return {
        "cell": cell,
        "n_all": int(len(ahs)),
        "n_full": int(full.sum()),
        "frac": float(full.mean()),
        "ah_full": float(np.median(ahs[full])) if full.any() else np.nan,
        "ah_part": float(np.median(ahs[~full])) if (~full).any() else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=None)
    args = ap.parse_args()

    base = Path(DATASETS["calce"]["dir"])
    zips = sorted(base.glob("*.zip"))
    if args.cells:
        want = set(args.cells)
        zips = [z for z in zips if z.stem in want]

    rows = []
    for i, zp in enumerate(zips, 1):
        r = one_zip(zp)
        if r is None:
            print(f"  [{i}/{len(zips)}] {zp.stem}: 解析失败", flush=True)
            continue
        rows.append(r)
        print(f"  [{i}/{len(zips)}] {r['cell']:8} all={r['n_all']:6} full={r['n_full']:6} "
              f"({r['frac']:.0%})  ah_full={r['ah_full']:.3f} ah_part={r['ah_part']:.3f}",
              flush=True)

    df = pd.DataFrame(rows).sort_values("frac", ascending=False)
    pd.set_option("display.width", 200)
    print("\n=== 汇总（按全深度占比排序）===")
    print(df.round(3).to_string(index=False))

    out = Path(__file__).resolve().parent / "data" / "processed" / "calce_depth.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("\n全深度占比 ≥ 50% 的电池：",
          df[df.frac >= 0.5].cell.tolist())
    print("全深度占比 < 20% 的电池：",
          df[df.frac < 0.2].cell.tolist())
    print("已保存", out)


if __name__ == "__main__":
    main()
