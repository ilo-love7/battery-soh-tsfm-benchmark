"""fewshot_finetune.py —— Chronos-Bolt 少样本**梯度微调**（真训练，不是校准）。

协议（与主实验口径一致）
------------------------
    支持集 support : MIT batch1（train，41 cells）取前 k 支
    测试集 test    : MIT batch2（test，43 cells）
    窗口           : context 512 点 + target 64 点，步长 16
    归一化         : 与评测完全一致 —— cap / median(cap[:5]) × 100
    损失           : Chronos-Bolt 自带的 quantile loss（模型内部算，直接 backward）

为什么只微调 Chronos-Bolt
-------------------------
其余三个（Chronos-T5 / TimesFM / Moirai）的 pip 包只暴露推理接口，没有可训练的
forward(loss)。Chronos-Bolt 的 `ChronosBoltModelForForecasting` 是标准 nn.Module，
forward(context, mask, target, target_mask) 会直接返回 loss，因此能做真梯度微调。
这一点必须在报告里写清楚：**few-shot 结论只对 Chronos-Bolt 成立**。

用法
----
    python fewshot_finetune.py --k 41 --steps 300
    python fewshot_finetune.py --k 4  --steps 300
    python fewshot_finetune.py --k 0                # 只做零样本基线（不训练）

产物
----
    shared/models/chronos-bolt-small-ft<k>/    微调后的权重（config.json 齐全，可被
                                               resolve_model_ref 直接识别）
    03_实验结果/fewshot/ft<k>_train.json       训练曲线与超参
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# torch 必须先于 pandas 导入（Windows 下 pyarrow 会遮蔽 c10.dll）
import torch  # noqa: I001

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs.config import DATASETS, MODELS_DIR, PROTOCOL, RESULTS_ROOT  # noqa: E402
from data.loaders import load_dataset  # noqa: E402
from data.protocol import clean_series, soh_normalize  # noqa: E402

BASE_MODEL = "amazon/chronos-bolt-small"
CTX = 512
HOR = 64
STRIDE = 16


def build_soh_series(cells: Dict[str, "object"]) -> Dict[str, np.ndarray]:
    """把每支电池的容量序列转成 SOH(%)，与评测的归一化口径完全一致。

    relative_split 里 q_ref 取自上下文开头的前 5 点中位数；由于上下文总是从序列
    起点开始（start >= 10），q_ref 等价于整条序列前 5 点的中位数 —— 所以这里
    可以一次性算好整条 SOH 曲线，不必逐起点重算。
    """
    out: Dict[str, np.ndarray] = {}
    for cid, df in cells.items():
        df = clean_series(df, PROTOCOL["outlier_thresh"])
        if len(df) < CTX + HOR:
            continue
        cap = df["capacity_Ah"].to_numpy(dtype=float)
        soh, _ = soh_normalize(cap, PROTOCOL["soh_ref_points"])
        if np.all(np.isfinite(soh)):
            out[cid] = soh.astype(np.float32)
    return out


def make_windows(series: Dict[str, np.ndarray], stride: int = STRIDE
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """滑窗采样：返回 (contexts[N,CTX], targets[N,HOR])。"""
    xs, ys = [], []
    for soh in series.values():
        n = len(soh)
        for s in range(CTX, n - HOR + 1, stride):
            xs.append(soh[s - CTX:s])
            ys.append(soh[s:s + HOR])
    if not xs:
        raise SystemExit("没有可用训练窗口：支持集序列太短")
    return np.stack(xs), np.stack(ys)


def load_base():
    """加载 Chronos-Bolt 原始权重（本地副本优先，走 resolve_model_ref）。"""
    from chronos import ChronosBoltPipeline
    from configs.config import resolve_model_ref

    ref = resolve_model_ref(BASE_MODEL)
    pipe = ChronosBoltPipeline.from_pretrained(ref, device_map="cpu", torch_dtype=torch.float32)
    return pipe


def finetune(xs: np.ndarray, ys: np.ndarray, steps: int, lr: float,
             batch: int, seed: int = 42, log_every: int = 25) -> Tuple[object, List[dict]]:
    pipe = load_base()
    model = pipe.model
    model.train()

    # 只训参数，不动 buffer；全参微调（48M，CPU 可承受）
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.1)

    rng = np.random.default_rng(seed)
    n = len(xs)
    hist: List[dict] = []
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = rng.integers(0, n, size=batch)
        ctx = torch.from_numpy(xs[idx])
        tgt = torch.from_numpy(ys[idx])
        mask = torch.ones_like(ctx, dtype=torch.bool)
        tmask = torch.ones_like(tgt, dtype=torch.bool)

        out = model(context=ctx, mask=mask, target=tgt, target_mask=tmask)
        loss = out.loss
        if loss is None or not torch.isfinite(loss):
            raise SystemExit(f"step {step}: loss 非法（{loss}）")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step == 1 or step % log_every == 0 or step == steps:
            rec = {"step": step, "loss": float(loss.item()),
                   "elapsed_sec": round(time.time() - t0, 1)}
            hist.append(rec)
            print(f"  step {step:4d}/{steps}  loss {rec['loss']:.4f}  "
                  f"{rec['elapsed_sec']:.0f}s", flush=True)

    model.eval()
    return pipe, hist


def save_pipe(pipe, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe.model.save_pretrained(str(out_dir))
    print(f"  权重写入 {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=41,
                    help="支持集电池数（0 = 只做零样本基线，不训练）")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="输出权重目录名后缀，默认 ft<k>")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[1/3] 加载支持集 MIT batch1 ...", flush=True)
    cells, _ = load_dataset("mit", DATASETS, split="train", with_meta=True)
    if not cells:
        raise SystemExit("支持集为空，检查 MIT 路径")
    ids = sorted(cells)[:args.k] if args.k > 0 else []
    support = build_soh_series({c: cells[c] for c in ids})
    print(f"      支持集 {len(support)} 支（要求 {args.k} 支，"
          f"序列长度 ≥ {CTX + HOR}）", flush=True)

    tag = args.out or f"ft{args.k}"
    rec = {"k": args.k, "steps": args.steps, "lr": args.lr, "batch": args.batch,
           "seed": args.seed, "ctx": CTX, "horizon": HOR, "stride": STRIDE,
           "base_model": BASE_MODEL, "support_cells": sorted(support)}

    if args.k == 0:
        print("      k=0：跳过训练，只记录零样本基线配置", flush=True)
        rec["history"] = []
        out_dir = None
    else:
        xs, ys = make_windows(support)
        rec["n_windows"] = int(len(xs))
        print(f"      训练窗口 {len(xs)} 个（ctx {CTX} + hor {HOR}，步长 {STRIDE}）", flush=True)
        print(f"[2/3] 微调 Chronos-Bolt：{args.steps} 步，lr {args.lr}，batch {args.batch}", flush=True)
        pipe, hist = finetune(xs, ys, args.steps, args.lr, args.batch, args.seed)
        rec["history"] = hist
        out_dir = MODELS_DIR / f"chronos-bolt-small-{tag}"
        save_pipe(pipe, out_dir)

    print("[3/3] 写训练记录", flush=True)
    rec_dir = RESULTS_ROOT / "fewshot"
    rec_dir.mkdir(parents=True, exist_ok=True)
    rec["model_id"] = f"amazon/chronos-bolt-small-{tag}" if out_dir else BASE_MODEL
    rec["output_dir"] = str(out_dir) if out_dir else None
    rec["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = rec_dir / f"{tag}_train.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)
    print(f"      -> {path}", flush=True)
    if out_dir:
        print(f"\n评测命令：\n  CHRONOS_MODEL_ID=amazon/chronos-bolt-small-{tag} \\\n"
              f"  python run_benchmark.py --model chronos-bolt --dataset mit --split test "
              f"--tag {tag}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
