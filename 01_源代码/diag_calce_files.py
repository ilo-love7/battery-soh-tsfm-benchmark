"""诊断 CALCE 原始 zip：逐文件判断"这个文件是不是容量老化测试"。

背景：CS2/CX2 每个 zip 里混着 SOC 循环、DST/FUDS、阻抗等多种测试，
`Discharge_Capacity(Ah)` 又是文件内累计值。若把 77 个文件当成一条寿命曲线，
容量会在 20%~130% 之间乱跳（实测轨迹图正是这样）。

用法：python diag_calce_files.py CS2_5
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from python_calamine import CalamineWorkbook

from configs.config import DATASETS

RAW = Path(DATASETS["calce"]["dir"])


def read_sheet(blob: bytes, sheet: str) -> pd.DataFrame:
    wb = CalamineWorkbook.from_filelike(io.BytesIO(blob))
    rows = wb.get_sheet_by_name(sheet).to_python()
    if not rows:
        return pd.DataFrame()
    head = [str(c).strip() if c is not None else "" for c in rows[0]]
    return pd.DataFrame(rows[1:], columns=head)


def main():
    cell = sys.argv[1] if len(sys.argv) > 1 else "CS2_5"
    z = zipfile.ZipFile(RAW / f"{cell}.zip")
    names = [n for n in z.namelist() if n.lower().endswith((".xlsx", ".xls"))]
    rows = []
    for n in names:
        blob = z.read(n)
        wb = CalamineWorkbook.from_filelike(io.BytesIO(blob))
        sh = [s for s in wb.sheet_names if s.startswith("Channel")]
        if not sh:
            continue
        df = read_sheet(blob, sh[0])
        if "Cycle_Index" not in df.columns or "Discharge_Capacity(Ah)" not in df.columns:
            continue
        df["Cycle_Index"] = pd.to_numeric(df["Cycle_Index"], errors="coerce")
        df["Discharge_Capacity(Ah)"] = pd.to_numeric(df["Discharge_Capacity(Ah)"], errors="coerce")
        df = df.dropna(subset=["Cycle_Index", "Discharge_Capacity(Ah)"])
        if df.empty:
            continue
        g = df.groupby("Cycle_Index")["Discharge_Capacity(Ah)"].max().sort_index()
        d = g.diff().dropna()
        d = d[d > 0]
        rows.append(dict(
            f=n.split("/")[-1], ncyc=int(df["Cycle_Index"].max()),
            cmin=int(df["Cycle_Index"].min()),
            med=round(float(d.median()), 3) if len(d) else np.nan,
            p10=round(float(d.quantile(0.1)), 3) if len(d) else np.nan,
            p90=round(float(d.quantile(0.9)), 3) if len(d) else np.nan,
        ))
    r = pd.DataFrame(rows).sort_values("med")
    pd.set_option("display.width", 200)
    print(f"=== {cell}: {len(r)} 个可解析文件 ===")
    print(r.to_string(index=False))
    print("\n按中位每循环放电量分档：")
    print(pd.cut(r["med"], [0, 0.3, 0.7, 0.95, 1.3, 1.6, 3, 1e9]).value_counts().sort_index())


if __name__ == "__main__":
    main()
