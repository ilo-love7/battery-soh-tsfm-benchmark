#!/usr/bin/env python3
"""文档一致性检查：相对链接是否存在、目录锚点是否有效。

断链与失效锚点是纯文档腐坏——没有任何单元测试会抓到它们，
而它们直接毁掉仓库的可读性（本项目历史上 README 的目录锚点全失效过一次）。

用法：
    python scripts/check_docs.py [文件...]
不带参数时检查 README.md 与 04_文档报告/*.md。

退出码：0 = 全部通过；1 = 有断链或失效锚点（会逐条列出）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def slug(heading: str) -> str:
    """GitHub 风格锚点：小写、空格转连字符、去掉标点（中文保留）。"""
    s = heading.strip().lower().replace(" ", "-")
    return re.sub(r"[^\w\u4e00-\u9fff-]", "", s)


def exists(base: Path, rel: str) -> bool:
    """链接目标是否存在；含通配符时按 glob 匹配。"""
    p = (base / rel).resolve()
    if "*" in rel or "?" in rel:
        return bool(list(Path(base).glob(rel)))
    return p.exists() or (ROOT / rel).resolve().exists()


def check(f: Path) -> tuple:
    text = f.read_text(encoding="utf-8")
    bad_links, bad_anchors = [], []
    for m in re.finditer(r"\]\((?!https?:|#|mailto:)([^)]+)\)", text):
        rel = m.group(1).split("#")[0].strip()
        if rel and not exists(f.parent, rel):
            bad_links.append(rel)
    heads = {slug(h) for h in re.findall(r"^#{1,6}\s+(.+?)\s*$", text, re.M)}
    for m in re.finditer(r"\]\(#([^)]+)\)", text):
        if m.group(1) not in heads:
            bad_anchors.append("#" + m.group(1))
    return bad_links, bad_anchors


def main(argv) -> int:
    if argv[1:]:
        files = [Path(a) for a in argv[1:]]
    else:
        files = [ROOT / "README.md", *sorted((ROOT / "04_文档报告").glob("*.md"))]
    n_link = n_anchor = 0
    for f in files:
        if not f.exists():
            print(f"!! 文件不存在：{f}")
            return 1
        bad_links, bad_anchors = check(f)
        for rel in bad_links:
            print(f"  断链  {f.name}: {rel}")
        for a in bad_anchors:
            print(f"  失效锚点 {f.name}: {a}")
        n_link += len(bad_links)
        n_anchor += len(bad_anchors)
    print(f"检查 {len(files)} 个文档：断链 {n_link} / 失效锚点 {n_anchor}")
    return 1 if (n_link or n_anchor) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
