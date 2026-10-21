"""真实数据加载器：MIT(MATR-1) / NASA(PCoE) / CALCE(UMD)。

统一返回 Dict[cell_id, pd.DataFrame]，DataFrame 至少含：
    cycle        int    循环号（从 1 起）
    capacity_Ah  float  该循环放电容量

设计原则：
- 只读原始文件，不复制、不改写（磁盘紧张，禁止把 20 GB 数据复制进项目目录）；
- 格式不符时显式报错，绝不静默返回占位数据；
- 各数据集真实格式（已逐一核实）：
    MIT   : stage-0 parquet（cell_id/cycle/QDischarge/SOH）——首选；
            MATR.zip 内每 cell 一个 pkl（cycle_data 含原始 V/I/T），仅备用；
    NASA  : .mat 结构体，m[cell][0,0]['cycle'][0] 为 616 个 cycle 结构，
            discharge 循环的 data['Capacity'] 即容量；
    CALCE : 每支电池一个 zip，内含按测试段命名的 xlsx；
            需按 Cycle Index 分组取放电容量。
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from configs.config import check_dataset  # noqa: E402  （config 不依赖 data，无循环导入）


# ---------------------------------------------------------------- MIT
def load_mit(cells_path: Path, cycles_path: Path,
             split: Optional[str] = None,
             with_meta: bool = False):
    """读取阶段 0 产出的 MIT 清洗数据。

    Args:
        split: None=全部；'train'/'test' 只取对应划分（沿用阶段 0 的划分）
        with_meta: True 时额外返回 cell 级元数据（含官方 cycle_life，
                   即 EOL 真值——本数据集序列已截断在 EOL 处，必须用它）
    """
    cells = pd.read_parquet(cells_path)
    cycles = pd.read_parquet(cycles_path)

    if split is not None:
        keep = set(cells.loc[cells["split"] == split, "cell_id"])
        cycles = cycles[cycles["cell_id"].isin(keep)]
        cells = cells[cells["cell_id"].isin(keep)]

    out: Dict[str, pd.DataFrame] = {}
    for cid, g in cycles.groupby("cell_id"):
        g = g.sort_values("cycle")
        df = pd.DataFrame({
            "cycle": g["cycle"].astype(int).values,
            "capacity_Ah": g["QDischarge"].astype(float).values,
        })
        df = df.dropna(subset=["capacity_Ah"])
        df = df[df["capacity_Ah"] > 0]          # 阶段 0 的 0 填充已被剔除，这里双保险
        out[str(cid)] = df.reset_index(drop=True)

    if with_meta:
        meta = cells.set_index("cell_id")[["cycle_life", "split", "batch", "q0"]].to_dict("index")
        return out, meta
    return out


# ---------------------------------------------------------------- NASA
def load_nasa(mat_dir: Path, cells: List[str]) -> Dict[str, pd.DataFrame]:
    """解析 NASA PCoE 的 .mat（每只电池一个文件）。"""
    import sys

    import scipy.io as sio

    out: Dict[str, pd.DataFrame] = {}
    missing = []
    for cid in cells:
        f = Path(mat_dir) / f"{cid}.mat"
        if not f.exists():
            # 不静默跳过：少了 cell 会让分母从 4 变成 3，而主表照样出数，
            # 下游只会看到"n=3"而不去想是文件没找到。这里必须告警。
            missing.append(cid)
            continue
        m = sio.loadmat(str(f), squeeze_me=True, struct_as_record=False)
        struct = m[cid]
        cycle_list = struct.cycle
        rows = []
        for c in np.atleast_1d(cycle_list):
            if getattr(c, "type", "") != "discharge":
                continue
            try:
                cap = float(c.data.Capacity)
            except (AttributeError, TypeError):
                continue
            rows.append(cap)
        if not rows:
            continue
        out[cid] = pd.DataFrame({
            "cycle": np.arange(1, len(rows) + 1),
            "capacity_Ah": rows,
        })
    if missing:
        print(f"  [nasa] ⚠️ 找不到 {len(missing)}/{len(cells)} 个 .mat：{missing}"
              f"（目录 {mat_dir}）。这些 cell 不会进入评测，分母会变小。",
              file=sys.stderr, flush=True)
    return out


# ---------------------------------------------------------------- CALCE
_CAP_COL_CANDIDATES = [
    "Discharge Capacity (Ah)",
    "Discharge Capacity(Ah)",
    "Capacity (Ah)",
    "Capacity(Ah)",
]
_CYCLE_COL_CANDIDATES = ["Cycle Index", "Cycle_Index", "Cycle", "cycle"]


def _pick_col(cols, candidates):
    for c in candidates:
        for actual in cols:
            if str(actual).strip().lower() == c.lower():
                return actual
    # 退化：包含关键字
    for actual in cols:
        low = str(actual).lower()
        if "discharge" in low and "capacity" in low:
            return actual
    return None


def _load_calce_xlsx(path: Path) -> Optional[pd.DataFrame]:
    """读一个 CALCE xlsx，返回 cycle/capacity_Ah。"""
    try:
        sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    except Exception:
        return None

    frames = []
    for _, df in sheets.items():
        if df is None or df.empty:
            continue
        cap_col = _pick_col(df.columns, _CAP_COL_CANDIDATES)
        cyc_col = _pick_col(df.columns, _CYCLE_COL_CANDIDATES)
        if cap_col is None or cyc_col is None:
            continue
        sub = df[[cyc_col, cap_col]].dropna()
        sub.columns = ["cycle", "capacity_Ah"]
        sub = sub[pd.to_numeric(sub["cycle"], errors="coerce").notna()]
        sub["cycle"] = sub["cycle"].astype(float).round().astype(int)
        sub["capacity_Ah"] = pd.to_numeric(sub["capacity_Ah"], errors="coerce")
        sub = sub.dropna()
        if not sub.empty:
            frames.append(sub)
    if not frames:
        return None
    allf = pd.concat(frames, ignore_index=True)
    # 同一循环可能有多段测试：取放电容量中位数（避免充电段混入）
    g = allf.groupby("cycle", as_index=False)["capacity_Ah"].median()
    return g.sort_values("cycle").reset_index(drop=True)


# 额定容量（Ah）：CS2 = 1.1 Ah LCO/石墨，CX2 = 1.35 Ah。用于判定"是否全深度老化数据"
_CALCE_RATED = {"CS2": 1.1, "CX2": 1.35}

# 满充满放的电压窗口：CALCE 单体上限 4.2 V、放电截止 2.7 V，留 0.05–0.15 V 容差。
# 实测：真正全深度老化的电芯（CX2_16 等）每个循环 Vmin≈2.70 / Vmax≈4.20；
# 浅工况文件（CS2_7 的 SOC 段）Vmin 只到 3.4–3.7，据此可稳定区分。
# ⚠️ 这两个数是对外承诺的口径，改了等于换数据集筛选，必须同步 README / 短文 /
#    02_数据集链接/CALCE_UMD.md 三处描述（2026-10-21 审计：文档曾写成 2.8/4.1，
#    与代码不一致；实测两套阈值选出的 14 支完全相同，仅 CX2_34 差 1 个循环，
#    而已发表数字由 2.85/4.15 产生，故统一改文档而非改代码）。
_V_HI = 4.15
_V_LO = 2.85

_CALCE_CACHE = Path(__file__).resolve().parent / "processed" / "calce_capacity_v3.parquet"
_CALCE_CACHE_META = _CALCE_CACHE.with_suffix(".meta.json")


def _calce_cache_key(zip_dir: Path, zips: List[Path], thresholds: tuple,
                     prefixes: Optional[List[str]] = None) -> dict:
    """缓存的"来源指纹"。

    ⚠️ 缓存必须绑定来源。曾出现过这样的隐患：load_calce 只看 `_CALCE_CACHE.exists()`
    就用缓存，完全不看 `CALCE_DIR`。别人把 CALCE_DIR 指到自己的一份 zip 上，
    拿到的却是本仓库作者机器上解析出来的 14 支——而且不报错。这是比"多跑十分钟"
    严重得多的正确性问题。

    ⚠️ `zips` 必须是**未按 prefixes 过滤的完整清单**。实测踩过：生产入口
    `load_dataset()` 会传 `prefixes=["CS2","CX2"]`（27 个 zip），而直接调
    `load_calce(dir)` 不带 prefixes 是 89 个 zip。若拿过滤后的清单算指纹，
    同一个目录两种调法算出两个指纹，每次交替调用都触发 10 分钟重解析。
    所以：`zips` 永远传全量，`prefixes` 单独作为一项写进指纹。
    """
    def _sig(p: Path):
        st = p.stat()
        return {"name": p.name, "size": st.st_size, "mtime": int(st.st_mtime)}
    # ⚠️ zip_dir 必须走一遍 Path() 规范化：直接 str() 的话，
    # `Path(...)` 会变成反斜杠、原始 str 保留正斜杠，同一目录算出两个指纹，
    # 每轮都误判"来源不一致"而触发 10–20 分钟的重复解析。
    return {"zip_dir": str(Path(zip_dir).expanduser()),
            "prefixes": sorted(prefixes) if prefixes else [],
            "zip_count": len(zips),
            "zips": [_sig(z) for z in zips],
            "v_lo": thresholds[0], "v_hi": thresholds[1],
            "min_full_frac": 0.9, "min_cycles": 100}


def _calce_cache_valid(zip_dir: Path, zips: List[Path], thresholds: tuple,
                       prefixes: Optional[List[str]] = None) -> bool:
    """缓存存在且来源指纹一致才允许复用。"""
    if not (_CALCE_CACHE.exists() and _CALCE_CACHE_META.exists()):
        return False
    try:
        old = json.loads(_CALCE_CACHE_META.read_text(encoding="utf-8"))
    except Exception:
        return False
    new = _calce_cache_key(zip_dir, zips, thresholds, prefixes)
    if old.get("zips") != new["zips"]:
        return False
    return all(old.get(k) == new[k] for k in ("zip_dir", "prefixes", "v_lo", "v_hi",
                                              "min_full_frac", "min_cycles"))


def load_calce(zip_dir: Path, prefixes: Optional[List[str]] = None,
               rebuild: bool = False, min_fade: float = -0.05,
               max_cycles: int = 20000, verbose: bool = False) -> Dict[str, pd.DataFrame]:
    """读取 CALCE 系列（每个 zip 一支电池，内部几十个 xlsx，解析一次后缓存）。

    缓存：data/processed/calce_capacity_v3.parquet（首次约 10–20 分钟，之后秒开）。
    缓存带"来源指纹"（全量 zip 清单 + 名字/大小/mtime + 电压阈值 + prefixes）：
    不一致就重新解析，绝不会拿旧缓存冒充新 `CALCE_DIR` 的数据。

    ⚠️ 缓存是**按本次调用覆盖写**，不是增量合并：`prefixes` 会决定解析哪些 zip，
    因此它必须进指纹。带 `["CS2","CX2"]` 和不带 prefixes 是两次不同的解析，
    交替调用会各跑一次——这是"正确但慢"，好过"快但拿到别人的数据"。

    ⚠️ 细胞筛选（写论文必须交代，详见 _calce_one_zip 文档串）：
    1. 循环级：只保留**电压窗口满充满放**的循环（Vmin ≤ 2.85 且 Vmax ≥ 4.15）；
    2. 电池级：全深度循环占比须 ≥ 90%，否则整支是浅工况/混合测试数据；
    3. 电池级：满放容量须在 [0.55, 1.25]×额定，否则不是单体全深度老化数据；
    4. 趋势级：末段/首段容量变化须 < min_fade（默认 −5%），剔除协议中途变更的电池；
    5. 循环数 ≤ max_cycles（CX2_3 有 9 万个微脉冲循环）。
    """
    thresholds = (_V_LO, _V_HI)
    zips_all = sorted(Path(zip_dir).glob("*.zip"))
    if prefixes:
        zips = [z for z in zips_all if any(z.name.startswith(p + "_") for p in prefixes)]
    else:
        zips = zips_all

    # 指纹永远基于**全量** zip 清单 + prefixes，与本次实际解析多少无关：
    # 否则"带 prefixes 的生产调用"和"不带 prefixes 的直接调用"会算出两个指纹，
    # 每次交替都触发重复解析（实测踩过）。
    if not rebuild and _calce_cache_valid(zip_dir, zips_all, thresholds, prefixes):
        allf = pd.read_parquet(_CALCE_CACHE)
        out = {}
        for cid, g in allf.groupby("cell_id"):
            out[str(cid)] = g[["cycle", "capacity_Ah"]].reset_index(drop=True)
        if prefixes:
            out = {k: v for k, v in out.items() if any(k.startswith(p + "_") for p in prefixes)}
        if out:
            if verbose:
                print(f"  [calce] 命中缓存 {_CALCE_CACHE.name}（来源指纹一致）")
            return _filter_calce(out, min_fade, max_cycles, verbose)

    if not zips:
        # 目录里没有可解析的 zip：不要打"重新解析 10–20 分钟"那种误导性提示，
        # 并区分"目录空"与"前缀不匹配"两种情况（两者修法完全不同）。
        all_zips = sorted(Path(zip_dir).expanduser().glob("*.zip"))
        if not all_zips:
            why = f"{Path(zip_dir).expanduser()} 下一个 .zip 都没有"
        else:
            why = (f"该目录有 {len(all_zips)} 个 zip，但没有任何文件名以 {prefixes} 开头")
        print(f"  [calce] {why}，返回 0 支。请检查 CALCE_DIR / prefixes 是否指对。")
        return {}

    if _CALCE_CACHE.exists() and not rebuild:
        print("  [calce] 缓存来源与当前 CALCE_DIR / 阈值不一致，重新解析原始 zip"
              "（约 10–20 分钟）。若要强行用旧缓存："
              f"删除 {_CALCE_CACHE_META.name} 后重跑，或设 rebuild 但请自行核对口径。")

    out: Dict[str, pd.DataFrame] = {}
    frames = []
    for i, zp in enumerate(zips, 1):
        rated = _CALCE_RATED.get(zp.stem.split("_")[0])
        df = _calce_one_zip(zp, rated=rated, verbose=verbose)
        if df is None or df.empty:
            if verbose:
                print(f"  [calce] {i}/{len(zips)} {zp.stem}: 不满足全深度老化条件，跳过",
                      flush=True)
            continue
        out[zp.stem] = df
        frames.append(df.assign(cell_id=zp.stem))
        print(f"  [calce] {i}/{len(zips)} {zp.stem}: {len(df)} 循环", flush=True)

    if frames:
        _CALCE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_parquet(_CALCE_CACHE, index=False)
        _CALCE_CACHE_META.write_text(
            json.dumps(_calce_cache_key(zip_dir, zips_all, thresholds, prefixes),
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"  [calce] 缓存写入 {_CALCE_CACHE}（含来源指纹 {_CALCE_CACHE_META.name}）")
    return _filter_calce(out, min_fade, max_cycles, verbose)


def _calce_sort_key(name: str, cell: Optional[str] = None):
    """按文件名里的日期排序。

    ⚠️ 文件名有两种布局，且**电池编号本身含数字**：
        CS2_5_10_01_12.xlsx         → 电池名在前，日期 10/01/12
        2_10_14_CX2_32_50cycle.xlsx → 日期在前，电池名在后

    早期实现用 `re.search(r"_(\\d{1,2})_(\\d{1,2})_(\\d{2})")` 取**第一个**匹配，
    "CS2_37_1_10_11" 被解析成 37 月 1 日（月份非法），排序整份失效，文件顺序被打乱，
    容量曲线因此出现"越用越高"的假象（实测 CS2_37 从 −79% 变成 +10%）。

    现在的做法：先剥掉电池名前缀，再按 `_` 切 token、滑窗取**第一个合法**的
    (月, 日, 两位年)，要求 月 1–12、日 1–31、年 08–20。
    """
    stem = name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if cell and stem.startswith(cell + "_"):
        stem = stem[len(cell) + 1:]

    toks = stem.split("_")
    for i in range(len(toks) - 2):
        a, b, c = toks[i], toks[i + 1], toks[i + 2]
        if not (a.isdigit() and b.isdigit() and c.isdigit()):
            continue
        if len(c) != 2:
            continue
        mo, dd, yy = int(a), int(b), int(c)
        if 1 <= mo <= 12 and 1 <= dd <= 31 and 8 <= yy <= 20:
            return (2000 + yy, mo, dd, stem)
    return (9999, 0, 0, stem)


def _calce_one_zip(zp: Path, rated: Optional[float] = None,
                   min_cycles: int = 100, min_full_frac: float = 0.9,
                   verbose: bool = False) -> Optional[pd.DataFrame]:
    """解析一个 CALCE zip（内含几十个按测试日期命名的 xlsx）。

    ⚠️ 三个实测坑（v2 骨架全踩了，逐条修）：
    1. 一个 zip 里有 **几十个 xlsx**（每次测试一个），不是只有一个——必须全读；
    2. `Discharge_Capacity(Ah)` 的语义在不同电池间**不一致**：多数是"文件内跨循环累计值"
       （CS2_5 第 100 循环 max=25.5 Ah），少数文件又是单循环值。直接当容量用会得到
       "容量随寿命上升"（实测 CS2_35 0.957→1.012）。
       → 改用**物理口径**电流积分 Q_cycle = Σ(-I·Δt)/3600，与计数器语义无关。
    3. **同一个 zip 里混着不同测试类型**（2026-10-20 补）：CS2_5 的 77 个文件里
       每循环放电量只有 0.22 Ah（额定 1.1 Ah），其实是 20% 深度 SOC 循环测试；
       CX2_3 是 DST 工况微脉冲（每循环 0.017 Ah）。

    第 4 次修正（2026-10-20 晚）：**用电压窗口判定"容量标定循环"**。
    之前靠"放电量接近 Q0"过滤，但一个循环放电量偏小既可能是浅工况，也可能是
    老化后期的正常衰减，两者分不开。物理判据是电压：
        满充满放 ⇔ Vmax ≥ 4.15 V 且 Vmin ≤ 2.85 V
    实测（`diag_calce_depth.py`）：旧口径的 14 支里，CS2_5/6/24/25 全深度占比只有
    **1.1%**、CS2_7 只有 9.9%——它们的主体是浅循环，不该进 SOH 基准；
    而 CX2_34/36/37/38、CS2_34/36/38 这些真正干净的电芯（占比 99.7%）反而被旧口径漏掉。

    现在的判据：
      a. **循环级**：只保留电压窗口满充满放的循环；
      b. **电池级**：全深度循环占比 ≥ min_full_frac（默认 0.9）——否则整支是浅工况数据；
      c. **电池级**：全深度循环放电量中位数须在 [0.55, 1.25]×额定（CX2_8 是 58 Ah 模组，剔除）；
      d. 保留循环数 ≥ min_cycles（默认 100）。
    """
    try:
        with zipfile.ZipFile(zp) as zf:
            names = [n for n in zf.namelist()
                     if n.lower().endswith((".xlsx", ".xls"))
                     and not n.startswith("__MACOSX") and not n.rsplit("/", 1)[-1].startswith("~$")]
            names.sort(key=lambda n: _calce_sort_key(n, zp.stem))

            frames: List[pd.DataFrame] = []
            for n in names:
                g = _calce_file_caps(zf, n)
                if g is not None and len(g) >= 2:
                    frames.append(g)
    except Exception:
        return None

    if not frames:
        return None

    allc = pd.concat(frames, ignore_index=True)

    # a. 循环级：电压窗口判定满充满放
    full = (allc["vmin"] <= _V_LO) & (allc["vmax"] >= _V_HI)
    frac = float(full.mean())

    # b. 电池级：全深度占比不足 → 整支是浅工况/混合测试数据
    if frac < min_full_frac:
        if verbose:
            print(f"    {zp.stem}: 全深度循环仅 {int(full.sum())}/{len(allc)} "
                  f"({frac:.1%}) < {min_full_frac:.0%}，剔除", flush=True)
        return None

    series = allc.loc[full, "capacity_Ah"].to_numpy(dtype=float)
    series = series[np.isfinite(series)]
    if len(series) < min_cycles:
        if verbose:
            print(f"    {zp.stem}: 全深度循环仅 {len(series)} < {min_cycles}，剔除", flush=True)
        return None

    q_med = float(np.median(series))

    # c. 电池级：容量量级须接近额定
    if rated:
        ratio = q_med / rated
        if not (0.55 <= ratio <= 1.25):
            if verbose:
                print(f"    {zp.stem}: 满放容量 {q_med:.3f} Ah = {ratio:.2f}×额定 {rated} Ah，"
                      f"非单体全深度数据，剔除", flush=True)
            return None

    # d. 循环级合理性：去掉明显截断/中断的循环（围绕中位数 0.5–1.3 倍）
    lo, hi = 0.5 * q_med, 1.3 * q_med
    keep = (series >= lo) & (series <= hi)
    series = series[keep]
    if len(series) < min_cycles:
        return None

    if verbose:
        print(f"    {zp.stem}: 全深度 {int(full.sum())}/{len(allc)} ({frac:.0%})，"
              f"Q={q_med:.3f} Ah，保留 {len(series)} 循环", flush=True)
    return pd.DataFrame({"cycle": np.arange(1, len(series) + 1),
                         "capacity_Ah": series})


def _calce_file_caps(zf: zipfile.ZipFile, name: str) -> Optional[pd.DataFrame]:
    """从单个 xlsx 提取每循环的放电量与电压窗口。

    返回 DataFrame[capacity_Ah, vmin, vmax]；电压列用于判定"满充满放"（见 _calce_one_zip）。

    用 calamine（Rust 解析器，17k 行 0.37s）读，openpyxl 兜底（同内容约 6s，慢 16 倍）。
    Δt 上限夹到 120 s：测试中断会留下巨大的时间跳变，不夹会产生天文数字容量。
    """
    import io

    try:
        raw = zf.read(name)
    except Exception:
        return None

    cols = ["Cycle_Index", "Current(A)", "Test_Time(s)", "Voltage(V)"]
    df = None
    try:
        sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None, engine="calamine")
        df = next((s for s in sheets.values()
                   if s is not None and set(cols) <= set(s.columns)), None)
    except Exception:
        df = None

    if df is None:                      # 兜底：openpyxl
        try:
            from openpyxl import load_workbook
            wb = load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
            ws = next((wb[s] for s in wb.sheetnames if s.lower().startswith("channel")),
                      wb[wb.sheetnames[-1]])
            rows = list(ws.iter_rows(values_only=True))
            wb.close()
            header = [str(c).strip() if c is not None else "" for c in rows[0]]
            idx = [header.index(c) for c in cols]
            df = pd.DataFrame([[r[i] for i in idx] for r in rows[1:]], columns=cols)
        except Exception:
            return None

    if df is None or df.empty:
        return None

    df = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        return None

    df = df.sort_values("Test_Time(s)")
    dt = df["Test_Time(s)"].diff().clip(lower=0, upper=120).fillna(0.0)
    current = df["Current(A)"]
    ah = (-current.where(current < -0.01, 0.0) * dt) / 3600.0
    g = pd.DataFrame({
        "capacity_Ah": ah.groupby(df["Cycle_Index"]).sum(),
        "vmin": df.groupby("Cycle_Index")["Voltage(V)"].min(),
        "vmax": df.groupby("Cycle_Index")["Voltage(V)"].max(),
    })
    g = g[g["capacity_Ah"] > 1e-4]
    return g if len(g) else None


def _filter_calce(out: Dict[str, pd.DataFrame], min_fade: float,
                  max_cycles: int, verbose: bool = False) -> Dict[str, pd.DataFrame]:
    """按容量趋势与循环数筛选可用的 CALCE 电池（理由见 load_calce 文档串）。"""
    keep, dropped = {}, []
    for cid, df in out.items():
        v = df["capacity_Ah"].to_numpy(dtype=float)
        n = len(v)
        if n > max_cycles:
            dropped.append((cid, n, float("nan"), "循环数异常"))
            continue
        if n < 100:
            dropped.append((cid, n, float("nan"), "序列过短"))
            continue
        k = max(1, min(20, n // 10))
        fade = float(v[-k:].mean() / v[:k].mean() - 1.0)
        if fade >= min_fade:
            dropped.append((cid, n, fade, "容量未下降（协议变更）"))
            continue
        keep[cid] = df
    if verbose and dropped:
        print(f"  [calce] 剔除 {len(dropped)} 支：")
        for cid, n, fade, why in dropped:
            f = "—" if fade != fade else f"{fade*100:+.1f}%"
            print(f"    {cid:8} n={n:6} 变化={f:>7}  {why}")
    return keep


# ---------------------------------------------------------------- 统一入口
def load_dataset(name: str, cfg: dict, split: Optional[str] = None,
                 with_meta: bool = False):
    name = name.lower()
    # 先做存在性检查：缺数据时给"该设哪个环境变量"的可读报错，
    # 而不是一路抛到 pandas 的 FileNotFoundError（干净克隆上必现）。
    check_dataset(name, cfg)
    if name == "mit":
        c = cfg["mit"]
        return load_mit(c["cells"], c["cycles"], split=split, with_meta=with_meta)
    if name == "nasa":
        c = cfg["nasa"]
        data = load_nasa(c["mat_dir"], c["cells"])
        return (data, {}) if with_meta else data
    if name == "calce":
        c = cfg["calce"]
        data = load_calce(c["dir"], c["prefixes"])
        return (data, {}) if with_meta else data
    raise ValueError(f"未知数据集: {name}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from configs.config import DATASETS

    for ds in ["mit", "nasa", "calce"]:
        try:
            data = load_dataset(ds, DATASETS)
            lens = {k: len(v) for k, v in list(data.items())[:5]}
            print(f"{ds}: {len(data)} cells, 前 5 长度 {lens}")
        except Exception as e:
            print(f"{ds}: 加载失败 -> {type(e).__name__}: {e}")
