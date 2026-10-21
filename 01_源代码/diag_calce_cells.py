"""批量判断每个 CALCE 电芯 zip 是不是"全深度容量老化"测试。

判据：文件内每循环放电量（累计值差分）的中位数是否接近额定容量。
    CS2 额定 1.1 Ah，CX2 额定 1.35 Ah
    ratio = med_delta / rated，>0.85 视为全深度循环

用法：python diag_calce_cells.py [--sample 12]
"""
from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from python_calamine import CalamineWorkbook

from configs.config import DATASETS

RAW = Path(DATASETS["calce"]["dir"])
RATED = {"CS2": 1.1, "CX2": 1.35}


def per_cycle_delta(blob: bytes) -> tuple[int, float] | None:
    wb = CalamineWorkbook.from_filelike(io.BytesIO(blob))
    sh = [s for s in wb.sheet_names if s.startswith("Channel")]
    if not sh:
        return None
    rows = wb.get_sheet_by_name(sh[0]).to_python()
    if not rows:
        return None
    head = [str(c).strip() if c is not None else "" for c in rows[0]]
    if "Cycle_Index" not in head or "Discharge_Capacity(Ah)" not in head:
        return None
    i_cyc, i_q = head.index("Cycle_Index"), head.index("Discharge_Capacity(Ah)")
    cyc, q = [], []
    for r in rows[1:]:
        try:
            cyc.append(float(r[i_cyc]))
            q.append(float(r[i_q]))
        except (TypeError, ValueError):
            continue
    if not cyc:
        return None
    df = pd.DataFrame({"c": cyc, "q": q})
    g = df.groupby("c")["q"].max().sort_index()
    d = g.diff().dropna()
    d = d[d > 0]
    if d.empty:
        return int(df["c"].max()), float("nan")
    return int(df["c"].max()), float(d.median())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=12, help="每个 zip 最多抽多少个文件")
    args = ap.parse_args()

    out = []
    for zp in sorted(RAW.glob("*.zip")):
        cell = zp.stem
        pref = cell.split("_")[0]
        if pref not in RATED:
            continue
        rated = RATED[pref]
        z = zipfile.ZipFile(zp)
        names = sorted(n for n in z.namelist() if n.lower().endswith((".xlsx", ".xls")))
        # 均匀抽样，避免只看开头（开头常是 SOC/形成测试）
        idx = np.linspace(0, len(names) - 1, min(args.sample, len(names))).round().astype(int)
        meds, ncyc = [], []
        for i in sorted(set(idx.tolist())):
            r = per_cycle_delta(z.read(names[i]))
            if r is None:
                continue
            ncyc.append(r[0])
            if not np.isnan(r[1]):
                meds.append(r[1])
        if not meds:
            out.append(dict(cell=cell, rated=rated, n_files=len(names),
                            med=np.nan, ratio=np.nan, full_frac=np.nan, max_cyc=np.nan))
            continue
        meds = np.array(meds)
        out.append(dict(
            cell=cell, rated=rated, n_files=len(names),
            med=round(float(np.median(meds)), 3),
            ratio=round(float(np.median(meds) / rated), 2),
            full_frac=round(float((meds / rated > 0.85).mean()), 2),
            max_cyc=int(max(ncyc)),
        ))
    df = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print(df.sort_values("ratio", ascending=False).to_string(index=False))
    print("\n判定为全深度老化（ratio>0.85 且 full_frac>=0.5）：")
    ok = df[(df["ratio"] > 0.85) & (df["full_frac"] >= 0.5)]
    print(ok["cell"].tolist(), f"共 {len(ok)} 支")
    print("\n判定为浅充放（ratio<=0.85）：")
    print(df[df["ratio"] <= 0.85]["cell"].tolist())


if __name__ == "__main__":
    main()
