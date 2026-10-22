#!/usr/bin/env python3
"""图一致性检查：把工作区里的图与 git 里已提交的版本逐张比对。

为什么要两种粒度：
  字节级 —— 完全相同的充要条件。同平台（同 OS + 同 FreeType/zlib）复现用它。
  像素级 —— 解码成数组再比。跨平台用：CI 跑在 Linux，而图是在 Windows 上
            生成的，FreeType / zlib 版本不同会让 PNG 字节变。
            像素不变 ⇒ 只是编码差异；像素也变 ⇒ 渲染栈不同或数据真的变了。

用法：
    python scripts/check_figures.py            # 默认要求字节级一致
    python scripts/check_figures.py --pixels   # 只要求像素级一致（跨平台）
    python scripts/check_figures.py --report   # 两种粒度都报，不判定

退出码：0 = 按所要求的粒度一致；1 = 不一致（或 pillow 缺失导致无法判定）。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "04_文档报告" / "figures"


def committed(rel: str) -> bytes:
    return subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT,
                          capture_output=True).stdout


def pixels(buf: bytes):
    """解码 PNG 为 numpy 数组；没装 pillow 时返回 None。"""
    try:
        import io

        import numpy as np
        from PIL import Image
    except ImportError:
        return None
    return np.asarray(Image.open(io.BytesIO(buf)).convert("RGB"))


def main(argv) -> int:
    mode = "bytes"
    if "--pixels" in argv:
        mode = "pixels"
    elif "--report" in argv:
        mode = "report"

    files = sorted(FIGDIR.glob("*.png"))
    if not files:
        print(f"!! {FIGDIR} 下没有 png")
        return 1

    n_byte_same = n_pix_same = 0
    rows = []
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        new = f.read_bytes()
        old = committed(rel)
        byte_same = (old == new)
        n_byte_same += byte_same
        po, pn = pixels(old), pixels(new)
        if po is None or pn is None or po.shape != pn.shape:
            pix_txt, pix_same = "（无法判定：缺 pillow 或尺寸不同）", None
        else:
            d = int((po != pn).any(axis=2).sum())
            mx = int(abs(po.astype(int) - pn.astype(int)).max())
            pix_same = (d == 0)
            n_pix_same += pix_same
            pix_txt = f"{d} 像素不同（最大差 {mx}）"
        rows.append((f.name, byte_same, pix_same, pix_txt))

    print(f"{'图':32s} {'字节':6s} {'像素':6s} 明细")
    for name, bs, ps, txt in rows:
        print(f"{name:32s} {'相同' if bs else '不同':6s} "
              f"{'相同' if ps else ('不同' if ps is False else '?'):6s} {txt}")
    print(f"\n共 {len(files)} 张：字节相同 {n_byte_same}，像素相同 {n_pix_same}")

    if mode == "report":
        return 0
    if mode == "bytes":
        ok = (n_byte_same == len(files))
        print("判定：字节级" + ("一致" if ok else "不一致"))
    else:
        ok = (n_pix_same == len(files))
        print("判定：像素级" + ("一致" if ok else "不一致"))
        if not ok:
            print("提示：像素不同可能是跨平台渲染栈差异，也可能是数据真的变了，"
                  "请结合 git diff 的文本产物一起看")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
