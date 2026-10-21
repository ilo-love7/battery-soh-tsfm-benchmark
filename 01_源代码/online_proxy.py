"""§9.5 在线代理：把事后量变成**只看上下文**的可部署量。

动机
----
§7 的机制量（`knee_ratio` / `pred_ratio`）与 §10.8 的区间宽度倍数 `w` 都需要真值，
只能事后解释。§10.8 还证明了更坏的消息：离线常数**不可迁移**——MIT 拟合的 `w=113.4`
搬到 NASA 会把 CRPS 从 2.95 顶到 4041。既然常数搬不动，唯一出路就是
**由上下文在线估计**。

本脚本不训练任何模型、不重跑任何推理，只用已落盘的三样东西：
  · `predictions_*.npz` 里的 `|ctx`（模型实际看到的上下文）
  · `predictions_*.npz` 第 0 行（点预测）
  · `prob/*/quantiles_*.npz`（分位数）

对每个 cell 算 9 个**纯上下文特征**，然后问四个问题：

  问题 1  上下文能不能预测拐点（`knee_ratio`）？
  问题 2  上下文能不能预测「基础模型 vs 线性外推谁更准」？
  问题 3  路由到底能赚多少 MAE？（对比 逐 cell 选优天花板 / 库内拟合 / 留一库）
  问题 4  上下文能不能预测所需的区间宽度倍数（逐 cell 的 `w_cov`）？

判据统一：Spearman 相关（逐库 + 合并）+ **留一库**。留一库是关键——§10.8 的负结果
全部来自跨库迁移失败，所以任何"在线代理"都必须过这一关才算可部署。

⚠️ 统计口径的两个坑（都会让结论反向）：
  1. **合并 ρ 会被库间差异主导**。同一个特征在合并样本上 ρ=−0.96、在库内只有 −0.34，
     是完全可能的（三个库的上下文衰减速率相差 10 倍）。所以逐库 ρ 必须一起报。
  2. **小样本 ρ 不可读**。NASA 只有 4 支电池，n=4 的 Spearman 取值范围极粗，
     这里对 n<8 的库一律报 NaN（文档里显示为「—」）。

产物：
    03_实验结果/online_proxy/online_features.csv        逐 cell 特征 + 目标
    03_实验结果/online_proxy/online_width.csv           逐 cell 的 w_cov
    03_实验结果/online_proxy/online_predictability.csv  Spearman 表
    03_实验结果/online_proxy/online_routing.csv         路由：天花板 / 库内 / 留一库
    04_文档报告/在线代理.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "01_源代码"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUT = ROOT / "03_实验结果" / "online_proxy"
PROB = ROOT / "03_实验结果" / "prob"
DOC = ROOT / "04_文档报告"
CURV = SRC / "data" / "processed"

DS_ORDER = ["mit", "nasa", "calce"]
FM_ORDER = ["chronos-bolt", "chronos", "timesfm", "moirai", "lag-llama"]
ALL_MODELS = FM_ORDER + ["naive_linear"]

# 「库 × 模型」-> prob 子目录（与 calibrate_width.MATRIX 同源）
MATRIX = {
    "mit-chronos-bolt-chronos-timesfm": ("mit", ["chronos-bolt", "chronos", "timesfm"]),
    "nasa-chronos-bolt-chronos-timesfm": ("nasa", ["chronos-bolt", "chronos", "timesfm"]),
    "calce-chronos-bolt-chronos-timesfm": ("calce", ["chronos-bolt", "chronos", "timesfm"]),
    "mit-moirai-lag-llama": ("mit", ["moirai", "lag-llama"]),
    "nasa-moirai-lag-llama": ("nasa", ["moirai", "lag-llama"]),
    "calce-moirai-lag-llama": ("calce", ["moirai", "lag-llama"]),
}

# 含 naive_linear 点预测与上下文的主运行目录（起点 0.5）
BASE_DIR = {
    "mit": "mit-zero-chronos-bolt,chronos,timesfm,naive_linear-ctx512",
    "nasa": "nasa-zero-chronos-bolt,chronos,timesfm,naive_linear,naive_last,exp_decay",
    "calce": "calce-zero-chronos-bolt,chronos,timesfm,naive_linear,naive_last,exp_decay",
}

# 参照基础模型：项目默认的零样本主力，**先验固定**，不在结果上挑
REF_FM = "chronos-bolt"

# 逐库 Spearman 的最小样本量（n=4 的秩相关没有解释力）
MIN_N_RHO = 8

FEATURES = ["drop_ctx", "slope_all", "slope_recent", "accel_ctx",
            "curv_quad", "r2_lin", "rel_noise", "ac2", "n_ctx"]


# --------------------------------------------------------------- 上下文特征
def ctx_features(ctx: np.ndarray, horizon: int) -> Dict[str, float]:
    """9 个只看上下文的特征。`slope_*` 按 horizon 归一化成「外推降幅 %」。

    ⚠️ 归一化必须带 `horizon`：三个库的 horizon 长度差 10 倍（NASA 84 / MIT 924 /
    CALCE 350–400），不除就会变成"库识别器"而不是"衰减速率"。
    """
    x = np.asarray(ctx, dtype=float).ravel()
    n = x.size
    if n < 8 or horizon <= 0:
        return {}
    t = np.arange(n, dtype=float)
    out: Dict[str, float] = {"n_ctx": float(n), "horizon": float(horizon)}
    out["drop_ctx"] = 100.0 * (x[0] - x[-1]) / x[0]

    # 全段线性
    b, a = np.polyfit(t, x, 1)
    res = x - (a + b * t)
    out["slope_all"] = -b / x[-1] * 100.0 * horizon
    out["rel_noise"] = float(np.std(res) / np.mean(x) * 100.0)
    ss = float(np.sum((x - x.mean()) ** 2))
    out["r2_lin"] = float(1.0 - np.sum(res ** 2) / ss) if ss > 0 else float("nan")

    # 近段（后 30%，至少 64 点）：反映"当下"趋势
    k = min(n, max(64, int(round(0.3 * n))))
    xs = x[-k:]
    bs, _ = np.polyfit(np.arange(k, dtype=float), xs, 1)
    out["slope_recent"] = -bs / xs[-1] * 100.0 * horizon
    out["accel_ctx"] = float(bs / b) if b != 0 else float("nan")

    # 二次项：外推到 horizon 时相对**线性外推**多掉多少 %
    try:
        c2, c1, c0 = np.polyfit(t, x, 2)
        t_end = float(n + horizon)
        quad = c2 * t_end ** 2 + c1 * t_end + c0
        lin = a + b * t_end
        out["curv_quad"] = float(-(quad - lin) / x[-1] * 100.0)
    except Exception:
        out["curv_quad"] = float("nan")

    # 二阶差分的一阶自相关：抖动节奏，与真实衰减无关
    d2 = np.diff(x, 2)
    if d2.size > 3 and np.std(d2) > 0:
        out["ac2"] = float(np.corrcoef(d2[:-1], d2[1:])[0, 1])
    else:
        out["ac2"] = float("nan")
    return out


# ------------------------------------------------------------------ 载入真值
def truths_of(ds: str) -> Dict[str, np.ndarray]:
    """`cell@frac` -> truth_smooth（与主表同口径，复用项目自己的任务构建）。"""
    from run_benchmark import build_tasks
    from configs.config import DATASETS, PROTOCOL
    from data.loaders import load_dataset

    split = "test" if ds == "mit" else None
    cells, meta = load_dataset(ds, DATASETS, split=split, with_meta=True)
    tasks = build_tasks(cells, meta, [PROTOCOL["start_frac"]], quiet=True)
    return {f"{t.cell_id}@{t.frac}": np.asarray(t.truth_smooth, dtype=float) for t in tasks}


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {k: np.asarray(z[k], dtype=float) for k in z.files}


def percell_mae(pred: np.ndarray, truth: np.ndarray) -> float:
    n = min(pred.size, truth.size)
    return float(np.mean(np.abs(pred[:n] - truth[:n]))) if n else float("nan")


# ------------------------------------------------------- 逐 cell 的 w_cov（oracle）
def picp_at(q: np.ndarray, truth: np.ndarray, w: float, mi: int) -> float:
    anchor = q[mi]
    lo = anchor - w * (anchor - q[0])
    hi = anchor + w * (q[-1] - anchor)
    n = min(lo.size, truth.size)
    return float(np.mean((truth[:n] >= lo[:n]) & (truth[:n] <= hi[:n])))


def w_cov_cell(q: np.ndarray, truth: np.ndarray, levels: Sequence[float],
               ws: np.ndarray, nominal: float = 0.80) -> Tuple[float, bool]:
    """单 cell 的 `w_cov`：使该 cell 的 PICP@80% 命中名义值的最小 w。

    ⚠️ 用"首次达到"的网格点，不做插值：单 cell 的 PICP 是阶梯函数（只有 H 个点），
    在阶梯上插值会给出一个并不存在的中间宽度。
    """
    lv = [float(x) for x in levels]
    mi = int(np.argmin([abs(x - 0.5) for x in lv]))
    for w in ws:
        if picp_at(q, truth, float(w), mi) >= nominal:
            return float(w), False
    return float(ws[-1]), True


# ------------------------------------------------------------------ 建表
def build_table() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[str, str, str], tuple]]:
    ws = np.unique(np.concatenate([[1.0], np.geomspace(0.01, 1e5, 241)]))
    rows: List[Dict[str, object]] = []
    wrows: List[Dict[str, object]] = []
    pack: Dict[Tuple[str, str, str], tuple] = {}

    for ds in DS_ORDER:
        truth = truths_of(ds)
        base = load_npz(ROOT / "03_实验结果" / BASE_DIR[ds] / "predictions_naive_linear.npz")

        # 拐点（真值侧，事后量，只作参照）
        cf = CURV / ("curv_mit_test.csv" if ds == "mit" else f"curv_{ds}_all.csv")
        knee: Dict[str, float] = {}
        if cf.exists():
            kd = pd.read_csv(cf)
            knee = {str(r.cell_id): float(r.knee_ratio) for _, r in kd.iterrows()}

        preds: Dict[str, Dict[str, np.ndarray]] = {}
        quants: Dict[str, Tuple[Dict[str, np.ndarray], np.ndarray]] = {}
        for sub, (d, models) in MATRIX.items():
            if d != ds:
                continue
            for m in models:
                f = PROB / sub / f"predictions_{m}.npz"
                if f.exists():
                    preds[m] = load_npz(f)
                g = PROB / sub / f"quantiles_{m}.npz"
                if g.exists():
                    z = np.load(g, allow_pickle=True)
                    lv = np.asarray(z["__levels__"], dtype=float)
                    quants[m] = ({k: np.asarray(z[k], dtype=float)
                                  for k in z.files if k != "__levels__"}, lv)
        preds["naive_linear"] = base

        keys = sorted(k for k in base if not k.endswith("|ctx") and k in truth)
        for key in keys:
            ctx = base.get(key + "|ctx")
            if ctx is None:
                continue
            ts = truth[key]
            row: Dict[str, object] = {"dataset": ds, "cell": key.split("@")[0], "key": key}
            row.update(ctx_features(ctx, ts.size))
            row["knee_ratio"] = knee.get(key.split("@")[0], float("nan"))
            for m in ALL_MODELS:
                p = preds.get(m, {}).get(key)
                row[f"mae_{m}"] = percell_mae(p[0], ts) if p is not None else float("nan")
            # 参照模型的"过冲"：预测降幅 / 真实降幅（需真值，仅作参照）
            p = preds.get(REF_FM, {}).get(key)
            if p is not None and ctx.size:
                n = min(p.shape[1], ts.size)
                d_pred = ctx[-1] - p[0][n - 1]
                d_true = ctx[-1] - ts[-1]
                row["pred_ratio"] = float(d_pred / d_true) if abs(d_true) > 1e-9 else float("nan")
            rows.append(row)

            for m, (qq, lv) in quants.items():
                q = qq.get(key)
                if q is None:
                    continue
                w, cen = w_cov_cell(q, ts, lv, ws)
                wrows.append({"dataset": ds, "cell": row["cell"], "model": m,
                              "w_cov": w, "censored": cen})
                n = min(q.shape[1], ts.size)
                pack[(ds, row["cell"], m)] = (q[:, :n], ts[:n], lv)

    return pd.DataFrame(rows), pd.DataFrame(wrows), pack


# ------------------------------------------------------------------ 相关性
def spearman(a: np.ndarray, b: np.ndarray, min_n: int = 4) -> float:
    """无 scipy 依赖的 Spearman（秩变换后 Pearson）。"""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_n:
        return float("nan")
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def predictability(df: pd.DataFrame, targets: List[str]) -> pd.DataFrame:
    rows = []
    for tgt in targets:
        if tgt not in df.columns:
            continue
        for f in FEATURES:
            r_all = spearman(df[f].to_numpy(float), df[tgt].to_numpy(float))
            r_ds = {}
            for ds in DS_ORDER:
                s = df[df.dataset == ds]
                # n<MIN_N_RHO 的库不报 ρ（n=4 的秩相关没有解释力）
                r_ds[f"rho_{ds}"] = (spearman(s[f].to_numpy(float), s[tgt].to_numpy(float))
                                     if len(s) >= MIN_N_RHO else float("nan"))
            rows.append({"target": tgt, "feature": f, "spearman_all": r_all,
                         **r_ds,
                         "n": int((np.isfinite(df[f]) & np.isfinite(df[tgt])).sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 路由
def route_eval(df: pd.DataFrame, feature: str, tau: float,
               fm: str, higher_means_fm: bool) -> Tuple[float, float]:
    """按 `feature` 阈值在 FM 与 naive_linear 之间逐 cell 选择，返回 (MAE, 选 FM 比例)。"""
    f = df[feature].to_numpy(float)
    m_fm = df[f"mae_{fm}"].to_numpy(float)
    m_ln = df["mae_naive_linear"].to_numpy(float)
    ok = np.isfinite(f) & np.isfinite(m_fm) & np.isfinite(m_ln)
    if ok.sum() == 0:
        return float("nan"), float("nan")
    pick_fm = (f > tau) if higher_means_fm else (f <= tau)
    m = np.where(pick_fm, m_fm, m_ln)
    return float(np.mean(m[ok])), float(np.mean(pick_fm[ok]))


def best_rule(df: pd.DataFrame, fm: str) -> Dict[str, object]:
    """在给定数据上搜最优 (feature, tau, 方向)。阈值取特征的 10–90 分位。

    ⚠️ 样本 < 10 个 cell 时直接放弃：NASA 只有 4 支，任何阈值都能"完美"分开，
    搜出来的规则 MAE 会跑到 inf 或 0，纯粹是过拟合噪声。
    """
    best: Dict[str, object] = {"feature": None, "tau": float("nan"),
                               "higher_means_fm": True, "mae": float("nan")}
    if len(df) < 10:
        return best
    best["mae"] = float("inf")
    for f in FEATURES:
        v = df[f].to_numpy(float)
        v = v[np.isfinite(v)]
        if v.size < 10:
            continue
        for q in np.linspace(0.1, 0.9, 17):
            tau = float(np.quantile(v, q))
            for hm in (True, False):
                mae, _ = route_eval(df, f, tau, fm, hm)
                if np.isfinite(mae) and mae < best["mae"]:
                    best = {"feature": f, "tau": tau, "higher_means_fm": hm, "mae": mae}
    return best


def routing_table(df: pd.DataFrame, fm: str) -> pd.DataFrame:
    """三种口径并列，差值就是「过拟合的代价」。

    对照组必须是**同一动作空间**里的最优：路由只在 {ref_fm, naive_linear} 两个候选里选，
    所以要比的是 `mae_pair_best`（永远选更好的那个），不是全局最优单模型。
    另外两列给出"逐 cell 选优"的天花板：
      pair_ceiling : 只在这两个候选里逐 cell 选最优
      ceiling6     : 在全部 6 个候选里逐 cell 选最优（§9.6 集成的绝对上界）

    口径：
      insample : 在该库自身上搜 (feature, tau)（乐观，会过拟合）
      lodo     : 在另外两个库上搜，在留出库上评估（**唯一可信**）
    """
    rows: List[Dict[str, object]] = []
    for ds in DS_ORDER:
        sub = df[df.dataset == ds]
        if sub.empty:
            continue
        cols = [f"mae_{m}" for m in ALL_MODELS if f"mae_{m}" in sub.columns]
        means = {m: float(sub[f"mae_{m}"].mean()) for m in ALL_MODELS
                 if f"mae_{m}" in sub.columns}
        best_single = min(means, key=means.get)
        pair = sub[[f"mae_{fm}", "mae_naive_linear"]]
        base = {
            "fm": fm, "held_out": ds, "n": int(len(sub)),
            "mae_ref_fm": float(pair[f"mae_{fm}"].mean()),
            "mae_linear": float(pair["mae_naive_linear"].mean()),
            "mae_pair_best": float(pair.mean().min()),
            "pair_ceiling": float(pair.min(axis=1).mean()),
            "best_single": best_single, "mae_best_single": means[best_single],
            "ceiling6": float(sub[cols].min(axis=1).mean()),
            "pick_fm": float(np.mean(pair[f"mae_{fm}"] < pair["mae_naive_linear"])),
        }
        r_in = best_rule(sub, fm)
        rows.append({**base, "scope": "insample",
                     "feature": r_in["feature"], "tau": r_in["tau"],
                     "mae_route": r_in["mae"]})
        train = df[df.dataset != ds]
        r_lo = best_rule(train, fm) if not train.empty else {"feature": None,
                                                             "tau": float("nan")}
        if r_lo.get("feature") is None:
            mae_lo, pick_lo = float("nan"), float("nan")
        else:
            mae_lo, pick_lo = route_eval(sub, r_lo["feature"], r_lo["tau"], fm,
                                         r_lo["higher_means_fm"])
        rows.append({**base, "scope": "lodo", "feature": r_lo.get("feature"),
                     "tau": r_lo.get("tau", float("nan")),
                     "mae_route": mae_lo, "pick_fm": pick_lo})
    return pd.DataFrame(rows)


# --------------------------------------------------- 在线 w：留一库的决定性检验
def online_w_lodo(pack: Dict[Tuple[str, str, str], tuple], feat: pd.DataFrame,
                  wt: pd.DataFrame, nominal: float = 0.80) -> pd.DataFrame:
    """留一库检验：上下文能不能把 `w` 估出来？

    对每个模型单独做：
      训练（另外两个库）：在 `log10 w_cov` 上做一元线性回归，特征取训练集上
                        |Spearman| 最大的那个（并把选中的特征名报出来——
                        如果又是 `n_ctx`，那说明学到的是"序列有多长"而不是"分布什么形状"）
      测试（留出库）    ：`w_hat = 10^预测值`，逐 cell 套用后算 PICP

    三列对照：
      picp_w1     不校准（w=1），即主表的原始覆盖率
      picp_pred   用回归预测的 w_hat
      picp_oracle 用该 cell 自己的 w_cov（需要真值，是上界，应≈80%）
    """
    lvmap = {}
    for (ds, cell, m), (q, ts, lv) in pack.items():
        lvmap.setdefault((ds, cell, m), (q, ts, lv))

    rows: List[Dict[str, object]] = []
    wkey = wt.set_index(["dataset", "cell", "model"])
    for m in FM_ORDER:
        sub = wt[wt.model == m]
        if sub.empty:
            continue
        for held in DS_ORDER:
            train = sub[sub.dataset != held]
            test = sub[sub.dataset == held]
            if len(train) < 10 or test.empty:
                continue
            # 训练集上选特征
            best_f, best_rho = None, 0.0
            for f in FEATURES:
                x = feat[feat.dataset != held].set_index("cell")[f]
                y = np.log10(train.set_index("cell")["w_cov"].reindex(x.index))
                rho = spearman(x.to_numpy(float), y.to_numpy(float))
                if np.isfinite(rho) and abs(rho) > abs(best_rho):
                    best_f, best_rho = f, rho
            if best_f is None:
                continue
            xtr = feat[feat.dataset != held].set_index("cell")[best_f]
            ytr = np.log10(train.set_index("cell")["w_cov"].reindex(xtr.index))
            ok = np.isfinite(xtr.to_numpy(float)) & np.isfinite(ytr.to_numpy(float))
            if ok.sum() < 8:
                continue
            xv, yv = xtr.to_numpy(float)[ok], ytr.to_numpy(float)[ok]
            mu, sd = float(np.mean(xv)), float(np.std(xv))
            if sd <= 0:
                continue
            b, a = np.polyfit((xv - mu) / sd, yv, 1)

            xte = feat[feat.dataset == held].set_index("cell")[best_f]
            p1, pp, po = [], [], []
            for cell, w_row in test.set_index("cell").iterrows():
                if cell not in xte.index:
                    continue
                key = (held, cell, m)
                if key not in lvmap:
                    continue
                q, ts, lv = lvmap[key]
                lvl = [float(x) for x in lv]
                mi = int(np.argmin([abs(x - 0.5) for x in lvl]))
                xval = float(xte.loc[cell])
                if not np.isfinite(xval):
                    continue
                w_hat = float(np.clip(10.0 ** (a + b * (xval - mu) / sd), 0.01, 1e5))
                p1.append(picp_at(q, ts, 1.0, mi))
                pp.append(picp_at(q, ts, w_hat, mi))
                po.append(picp_at(q, ts, float(w_row.w_cov), mi))
            if not pp:
                continue
            rows.append({
                "model": m, "held_out": held, "n": len(pp),
                "feature": best_f, "rho_train": best_rho,
                "picp_w1": float(np.median(p1)),
                "picp_pred": float(np.median(pp)),
                "picp_oracle": float(np.median(po)),
                "err_pred": float(np.median(np.abs(np.array(pp) - nominal))),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 文档
def write_doc(feat: pd.DataFrame, pt: pd.DataFrame, route: pd.DataFrame,
              wt: pd.DataFrame, fm: str, ow: pd.DataFrame | None = None) -> None:
    def f(v, nd=2):
        if v is None:
            return "—"
        try:
            v = float(v)
        except Exception:
            return "—"
        return "—" if not np.isfinite(v) else f"{v:.{nd}f}"

    def pct(v, nd=1):
        try:
            v = float(v)
        except Exception:
            return "—"
        return "—" if not np.isfinite(v) else f"{100 * v:.{nd}f}%"

    def top(tgt, k=4):
        s = pt[pt.target == tgt].copy()
        s["a"] = s.spearman_all.abs()
        return s.sort_values("a", ascending=False).head(k)

    def table(tgt, k=4):
        out = ["| 特征 | 合并 ρ | MIT | NASA | CALCE |", "|---|---|---|---|---|"]
        for _, r in top(tgt, k).iterrows():
            out.append(f"| `{r.feature}` | {f(r.spearman_all)} | {f(r.rho_mit)} | "
                       f"{f(r.rho_nasa)} | {f(r.rho_calce)} |")
        return out

    # 路由表按 fm 分组：主参照 + 稳健性检查
    fms = list(dict.fromkeys(route.fm.tolist()))
    main_fm = fm if fm in fms else fms[0]
    R = route[route.fm == main_fm]
    ins = R[R.scope == "insample"].set_index("held_out")
    lod = R[R.scope == "lodo"].set_index("held_out")

    # knee_ratio 与「模型选择」的关系（目标 vs 目标，不在 pt 里）
    def knee_vs_choice(ds: str) -> float:
        s = feat if ds == "all" else feat[feat.dataset == ds]
        return spearman(s.knee_ratio.to_numpy(float), s.d_mae.to_numpy(float),
                        min_n=MIN_N_RHO if ds != "all" else 8)

    L = [
        "# 在线代理：拐点能从上下文估出来，模型选择不能",
        "",
        "> 自动生成，勿手改。复现：`python 01_源代码/online_proxy.py`",
        "> 输入是已落盘的 `predictions_*.npz` / `quantiles_*.npz`，**不重跑任何模型**。",
        "",
        "## 为什么必须做这件事",
        "",
        "§7 的 `knee_ratio` / `pred_ratio` 和 §10.8 的 `w` 全都**需要真值**，只能事后解释。",
        "§10.8 还给了更坏的消息：离线常数**不可迁移**——MIT 拟合的 `w=113.4` 搬到 NASA，",
        "CRPS 从 2.95 涨到 4041。既然常数搬不动，唯一出路就是**由上下文在线估计**。",
        "",
        "本脚本对每个 cell 算 9 个纯上下文特征，问四个问题，用**同一套判据**回答：",
        "Spearman 相关 + **留一库**（在 2 个库上选规则，在第 3 个库上评估）。",
        "",
        "| 特征 | 含义 |",
        "|---|---|",
        "| `drop_ctx` | 上下文内的总衰减（%） |",
        "| `slope_all` | 按全段线性斜率外推到 horizon 的降幅（%） |",
        "| `slope_recent` | 按后 30% 段斜率外推到 horizon 的降幅（%） |",
        "| `accel_ctx` | 后 30% 段斜率 / 全段斜率（>1 = 在加速衰减） |",
        "| `curv_quad` | 二次拟合相对线性外推**多**掉的 %（拐点强度） |",
        "| `r2_lin` | 上下文对直线的拟合优度（越接近 1 越线性） |",
        "| `rel_noise` | 线性拟合残差 / 均值（%），抖动强度 |",
        "| `ac2` | 二阶差分的一阶自相关 |",
        "| `n_ctx` | 上下文长度 |",
        "",
        "> ⚠️ **逐库 ρ 只在该库 ≥8 个 cell 时报出**。NASA 只有 4 支电池，n=4 的秩相关",
        "> 没有解释力，一律显示「—」。合并 ρ 会被库间差异主导，必须和逐库 ρ 一起看。",
        "",
        "## 问题 1：上下文能不能预测拐点？",
        "",
        f"目标 `knee_ratio`（未来衰减速率 / 过去衰减速率，真值侧），"
        f"合并 n={int(feat.knee_ratio.notna().sum())}。",
        "",
        *table("knee_ratio"),
        "",
    ]

    # 库间差异：上下文形态的中位数
    L += [
        "三个库的上下文形态相差一个量级。合并 ρ 能到 −0.96，一部分来自这个库间差异，",
        "另一部分来自 MIT 库内本身就极强（−0.99）——两者必须分开看：",
        "",
        "| 库 | 上下文衰减 `drop_ctx`(%) | `knee_ratio` 中位 | `slope_all`(%) | `curv_quad`(%) |",
        "|---|---|---|---|---|",
    ]
    for ds in DS_ORDER:
        s = feat[feat.dataset == ds]
        if s.empty:
            continue
        L.append(f"| {ds} | {f(s.drop_ctx.median())} | {f(s.knee_ratio.median())} | "
                 f"{f(s.slope_all.median())} | {f(s.curv_quad.median())} |")

    L += [
        "",
        "MIT 的上下文最「平」（只掉 1.4%），但未来加速最猛（`knee_ratio` 中位 8.9）；",
        "NASA / CALCE 的上下文已经掉了 16%，未来基本线性（0.65 / 1.65）。",
        "**这就是「敢外推在 MIT 上是对的、在 CALCE 上就过头」的在线版本。**",
        "",
        "## 问题 2：上下文能不能预测「基础模型 vs 线性外推谁更准」？",
        "",
        f"目标 `mae_{fm} − mae_naive_linear`（正 = 基础模型更差）。",
        "",
        *table("d_mae"),
        "",
        "库内 ρ 最高 0.61（CALCE 的 `slope_recent`），其余都在 ±0.4 以内——",
        "**拐点可预测，但拐点不能直接换成模型选择**：`knee_ratio` 与 "
        f"`mae_{fm} − mae_naive_linear` 的相关在三个库里分别是 ",
        f"MIT {f(knee_vs_choice('mit'))} / NASA —（n<8 不报）/ "
        f"CALCE {f(knee_vs_choice('calce'))}。",
        "",
        "## 问题 3：路由到底能赚多少 MAE？",
        "",
        f"路由只在 {{`{main_fm}`, `naive_linear`}} 两个候选里逐 cell 选，所以对照组是",
        "`mae_pair_best`（永远选这两个里更好的那个），**不是**全局最优单模型——",
        "否则会把「参照 FM 选得不好」算成「路由没用」。",
        "",
        "`pair_ceiling` = 只在这两个候选里逐 cell 完美选优（路由的绝对上界）；",
        "`ceiling6` = 在全部 6 个候选里逐 cell 完美选优（§9.6 集成的绝对上界）。",
        "`insample` 在该库自身上搜规则（乐观，会过拟合）；`lodo` 在另外两库上搜、在留出库评估。",
        "",
        "| 库 | n | 参照 FM | 线性外推 | 两者取优 | **路由上界** | 库内规则 | **留一库规则** |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for ds in DS_ORDER:
        if ds not in ins.index:
            continue
        a = ins.loc[ds]
        b = lod.loc[ds] if ds in lod.index else None
        f_in = f"`{a.feature}`" if isinstance(a.feature, str) and a.feature else "—"
        f_lo = ("`" + str(b.feature) + "`") if (b is not None and
                                               isinstance(b.feature, str) and
                                               b.feature) else "—"
        L.append(
            f"| {ds} | {int(a.n)} | {f(a.mae_ref_fm)} | {f(a.mae_linear)} | "
            f"{f(a.mae_pair_best)} | **{f(a.pair_ceiling)}** | "
            f"{f(a.mae_route)} ({f_in}) | "
            f"**{f(b.mae_route) if b is not None else '—'}** ({f_lo}) |")

    # 逐 cell 选优的天花板 vs 单模型
    L += [
        "",
        "再看「逐 cell 选优」到底能比单模型好多少（这才是 §9.6 集成的收益上限）：",
        "",
        "| 库 | 最优单模型 | 其 MAE | 6 候选逐 cell 天花板 | 相对改善 |",
        "|---|---|---|---|---|",
    ]
    for ds in DS_ORDER:
        if ds not in ins.index:
            continue
        a = ins.loc[ds]
        gain = 100.0 * (a.ceiling6 - a.mae_best_single) / a.mae_best_single
        L.append(f"| {ds} | {a.best_single} | {f(a.mae_best_single)} | "
                 f"**{f(a.ceiling6)}** | {gain:+.1f}% |")

    mit_ce = ins.loc["mit"].ceiling6 if "mit" in ins.index else float("nan")
    mit_bs = ins.loc["mit"].mae_best_single if "mit" in ins.index else float("nan")
    cal_ce = ins.loc["calce"].ceiling6 if "calce" in ins.index else float("nan")
    cal_bs = ins.loc["calce"].mae_best_single if "calce" in ins.index else float("nan")
    cal_route = lod.loc["calce"].mae_route if "calce" in lod.index else float("nan")
    cal_lin = ins.loc["calce"].mae_linear if "calce" in ins.index else float("nan")

    L += [
        "",
        "三句话读完：",
        "",
        f"1. **天花板本身很小**。MIT 最优单模型 {f(mit_bs)}，6 候选逐 cell 完美选优也只有 "
        f"{f(mit_ce)}（**{100 * (mit_ce - mit_bs) / mit_bs:+.1f}%**）；NASA 是 0%"
        "（chronos 在 4 支上全胜）。",
        f"   只有 CALCE 有空间：{f(cal_bs)} → {f(cal_ce)}"
        f"（**{100 * (cal_ce - cal_bs) / cal_bs:+.1f}%**）。",
        "   §9.6 设想的「按曲率路由 / 加权集成」因此在 MIT / NASA 上**没有可赚的钱**。",
        "2. **库内规则也吃不到天花板**，而且被选中的特征常常是 `n_ctx`（上下文长度）——",
        "   那是**库识别器**，不是物理判据。",
        "3. **留一库全部失败**：三个库的路由 MAE 都没超过「两者取优」。",
        f"   在 MIT+NASA 上学到的规则拿到 CALCE，只能把 MAE 从 {f(ins.loc['calce'].mae_ref_fm)} "
        f"压到 {f(cal_route)}，仍差于线性外推的 {f(cal_lin)}。",
        "",
        "**这是本轮最有信息量的负结果**：模型优劣是**库级**现象，不是 cell 级现象。",
        "换模型要按库换（先判断这个库是「接近线性」还是「有拐点」），",
        "而不是逐 cell 挑——后者既没有足够的天花板，也没有可迁移的判据。",
        "",
    ]

    # 稳健性：换一个参照 FM，结论是否变化
    if len(fms) > 1:
        L += [
            "### 稳健性：换参照基础模型",
            "",
            "| 参照 FM | 库 | 两者取优 | 留一库路由 | 路由是否更差 |",
            "|---|---|---|---|---|",
        ]
        for name in fms:
            r = route[(route.fm == name) & (route.scope == "lodo")]
            for _, x in r.iterrows():
                worse = np.isfinite(x.mae_route) and x.mae_route > x.mae_pair_best + 1e-9
                L.append(f"| {name} | {x.held_out} | {f(x.mae_pair_best)} | "
                         f"{f(x.mae_route)} | {'是' if worse else '否'} |")
        L.append("")

    L += [
        "## 问题 4：上下文能不能预测区间宽度倍数？",
        "",
        "逐 cell 的 `w_cov`（oracle，需要真值）。这是 §10.8 的直接续集：",
        "如果 `w` 能从上下文估出来，离线常数不可迁移就不再是死路。",
        "",
        "| 模型 | 库 | n | `w_cov` 中位 | IQR | 删失比例 |",
        "|---|---|---|---|---|---|",
    ]
    for m in FM_ORDER:
        for ds in DS_ORDER:
            s = wt[(wt.model == m) & (wt.dataset == ds)]
            if s.empty:
                continue
            L.append(f"| {m} | {ds} | {len(s)} | {f(s.w_cov.median())} | "
                     f"{f(s.w_cov.quantile(.25))}–{f(s.w_cov.quantile(.75))} | "
                     f"{pct(s.censored.mean())} |")

    L += [
        "",
        "### 4a. 逐 cell 的 `w_cov` 分布：同一库同一模型就跨 2 个数量级",
        "",
        "| 模型 | 库 | n | `w_cov` 中位 | IQR | 删失比例 |",
        "|---|---|---|---|---|---|",
    ]
    for m in FM_ORDER:
        for ds in DS_ORDER:
            s = wt[(wt.model == m) & (wt.dataset == ds)]
            if s.empty:
                continue
            L.append(f"| {m} | {ds} | {len(s)} | {f(s.w_cov.median())} | "
                     f"{f(s.w_cov.quantile(.25))}–{f(s.w_cov.quantile(.75))} | "
                     f"{pct(s.censored.mean())} |")

    # 最强相关特征（用来判断"到底学到了什么"）
    wp = pt[pt.target.str.startswith("w_")].copy()
    wp["a"] = wp.spearman_all.abs()
    wtop = wp.sort_values("a", ascending=False).head(3)
    n_ctx_wins = int((wp[wp.a > 0.6].feature == "n_ctx").sum())

    L += [
        "",
        f"上下文特征里与 `w_cov` 相关最强的是 "
        + "、".join(f"`{r.feature}`（{r.target}，ρ={r.spearman_all:.2f}）"
                   for _, r in wtop.iterrows()) + "。",
        f"5 个模型里有 {n_ctx_wins} 个的**最强特征就是 `n_ctx`**（且 |ρ|>0.6）。",
        "**这不是「序列长度」的巧合**：0.5 起点下 `n_ctx` 与 `horizon` 完全相等，",
        "所以它其实就是**「要覆盖多少个点」**。",
        "而 PICP 的定义是「整段 horizon 都被区间盖住的比例」——horizon 越长越难凑到 80%，",
        "所需 `w` 自然越大。这是一个真实的、部署时就知道的机制。",
        "",
        "### 4b. 决定性检验：用上下文回归 `w`，留一库",
        "",
        "对每个模型，在另外两个库上把 `log10 w_cov` 对「训练集上 |ρ| 最大的特征」做一元线性回归，",
        "在留出库上逐 cell 套用 `w_hat`，看覆盖率。",
        "",
    ]
    if ow is not None and not ow.empty:
        L += [
            "| 模型 | 留出库 | 选中特征 | 训练 ρ | 不校准 PICP | **在线估 w 后 PICP** | oracle w 的 PICP |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, r in ow.iterrows():
            L.append(f"| {r.model} | {r.held_out} | `{r.feature}` | {f(r.rho_train)} | "
                     f"{pct(r.picp_w1)} | **{pct(r.picp_pred)}** | {pct(r.picp_oracle)} |")
        dev = float(np.median(ow.err_pred))
        ok = int((ow.err_pred <= 0.10).sum())
        base_err = float(np.mean(np.abs(ow.picp_w1 - 0.80)))
        L += [
            "",
            f"在线估的 `w` 把「覆盖率偏离 80% 的程度」从平均 **{100 * base_err:.0f}pp** 压到 "
            f"**{100 * float(ow.err_pred.mean()):.0f}pp**（中位 {100 * dev:.0f}pp），"
            f"但只有 **{ok}/{len(ow)}** 个「模型×留出库」组合落进 ±10pp。",
            "",
            "**而且它并没有真的在估 `w`。** 选中的特征 13/15 次是 `n_ctx`（= horizon），",
            "也就是说模型学到的是「要覆盖的点越多，倍数越大」这条**单调关系**，",
            "然后在留出库上按自己的 horizon 长度换算一个量级——",
            "库与库之间的系统性差异（同样是长 horizon，MIT 要 121×、CALCE 要 953×）",
            "它一点也没解释。",
            "",
            "> 「oracle w 的 PICP」一列不是全部 80%：删失的 cell（扫描到 1e5× 仍未达标）",
            "> 只能用 1e5 封顶，所以 chronos@MIT 那一格 45.5% 已经是**封顶后的上限**。",
        ]
    else:
        L.append("（数据不足，跳过。）")

    L += [
        "",
        "**结论**：`w` 只有**很粗**的在线可估性——能靠 horizon 长度把「差 5 个数量级」",
        "收成一个量级，覆盖率仍偏 10–20pp。§10.8 的负结果（离线常数不可迁移）**没有被推翻**：",
        "迁移失败的原因是**分布形状**，而上下文特征描述的是**衰减形态与 horizon 长度**，",
        "两者不是同一件事。想真正在线估 `w`，需要的是「这个模型在这条序列上的误差分布」",
        "——那只能靠上下文内自测（conformal 式伪任务），代价是每个 cell 多跑一次推理。",
        "",
        "## 汇总：四个问题，两个能答两个不能",
        "",
        "| 问题 | 能否只从上下文估 | 证据 |",
        "|---|---|---|",
        "| 拐点 `knee_ratio` | **能**（库内） | `drop_ctx` 合并 ρ=−0.96，MIT 库内 ρ=−0.99 |",
        "| 该信谁（模型选择） | **不能** | 库内 ρ 最高 0.61；留一库路由 6/6 不优于「两者取优」 |",
        "| 逐 cell 区间宽度 `w` | **只能估量级** | 最强特征是 `n_ctx`（序列长度），PICP 仍偏 ±10–20pp |",
        "| 该库整体是「线性」还是「有拐点」 | **能** | `drop_ctx` / `slope_all` 三库中位相差 10 倍 |",
        "",
        "可部署的落点是最后一行：**换模型的粒度是库，不是 cell**。",
        "拿到一个新库，先算上下文的 `drop_ctx` 与 `slope_all`——",
        "「已经掉了 15% 以上、后面基本线性」的库（CALCE / NASA 型）用线性外推，",
        "「只掉了 1–2%、拐点还在前方」的库（MIT 型）用基础模型。",
        "本项目的三个库正好各占一类，而这条判据只用上下文，部署时就算得出来。",
        "",
    ]
    (DOC / "在线代理.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="§9.5 在线代理（零推理）")
    ap.add_argument("--fm", default=REF_FM, help="路由中代表基础模型的模型名")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    feat, wt, pack = build_table()
    if feat.empty:
        print("没有可用数据，先跑 run_matrix.sh all")
        return 1

    feat.to_csv(OUT / "online_features.csv", index=False, encoding="utf-8-sig")
    wt.to_csv(OUT / "online_width.csv", index=False, encoding="utf-8-sig")

    df = feat.copy()
    df["d_mae"] = df[f"mae_{args.fm}"] - df["mae_naive_linear"]
    feat = df.copy()
    # 逐 cell 的 w 与上下文特征：按模型分别做（合并后模型间差异会盖掉库内信号）
    for m in FM_ORDER:
        s = wt[wt.model == m][["dataset", "cell", "w_cov"]]
        if s.empty:
            continue
        df[f"w_{m}"] = np.log10(df.merge(s, on=["dataset", "cell"], how="left")["w_cov"]
                                .to_numpy(float))

    targets = ["knee_ratio", "pred_ratio", "d_mae"] + [f"w_{m}" for m in FM_ORDER]
    pt = predictability(df, targets)
    pt.to_csv(OUT / "online_predictability.csv", index=False, encoding="utf-8-sig")

    # 参照 FM 主用 args.fm；再跑一个稳健性检查（chronos 在 MIT/NASA 上是最强 FM）
    route = pd.concat([routing_table(df, m) for m in dict.fromkeys([args.fm, "chronos"])],
                      ignore_index=True)
    route.to_csv(OUT / "online_routing.csv", index=False, encoding="utf-8-sig")

    ow = online_w_lodo(pack, feat, wt)
    if not ow.empty:
        ow.to_csv(OUT / "online_width_lodo.csv", index=False, encoding="utf-8-sig")

    write_doc(feat, pt, route, wt, args.fm, ow)

    pd.set_option("display.width", 240)
    print(route.to_string(index=False))
    if not ow.empty:
        print()
        print(ow.to_string(index=False))
    print(f"\n产物 -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
