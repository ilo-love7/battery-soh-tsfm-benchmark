"""CALCE 缓存"来源指纹"的回归测试。

无第三方依赖（不需要 pytest），直接 `python tests/test_cache_fingerprint.py` 跑。
锁死的是 2026-10-21 第十轮复审里连续踩到的四个坑：

1. `load_calce` 曾在文件里定义两遍，新版本被旧版本覆盖、指纹一度不生效；
2. 指纹用 `str(zip_dir)`，`Path` 与原始 `str` 算出两个指纹 → 每轮重复解析；
3. 指纹在 `prefixes` 过滤**之后**算，生产调用与直接调用交替触发重复解析；
4. 缓存只看 `exists()`，换目录也返回旧数据（最严重的正确性问题）。
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.loaders import (  # noqa: E402
    _CALCE_CACHE,
    _CALCE_CACHE_META,
    _calce_cache_key,
    _calce_cache_valid,
    load_calce,
)

_PASS: list[str] = []
_FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (_PASS if cond else _FAIL).append(name)
    print(f"  {'✓' if cond else '✗'} {name}" + (f"  — {detail}" if detail and not cond else ""))


def _mk_zips(d: Path, n: int = 3) -> list[Path]:
    """造 n 个假 zip（内容无所谓，只用于算指纹）。"""
    out = []
    for i in range(n):
        p = d / f"CS2_{i}_10_01_12.zip"
        p.write_bytes(b"PK\x03\x04fake")
        out.append(p)
    return sorted(out)


def test_key_is_path_form_independent() -> None:
    """坑 2：Path 与 str 必须算出同一个指纹。"""
    print("\n[1] 指纹与路径书写形式无关")
    with tempfile.TemporaryDirectory() as td:
        zips = _mk_zips(Path(td))
        a = _calce_cache_key(Path(td), zips, (2.85, 4.15), ["CS2", "CX2"])
        b = _calce_cache_key(td, zips, (2.85, 4.15), ["CS2", "CX2"])
        # 正斜杠 vs 反斜杠：Windows 上 Path() 会规范化，str() 不会
        check("Path 与 str 的 zip_dir 相同", a["zip_dir"] == b["zip_dir"],
              f"{a['zip_dir']!r} != {b['zip_dir']!r}")
        check("整个指纹相同", a == b)


def test_key_uses_full_zip_list() -> None:
    """坑 3：指纹必须基于全量 zip，且 prefixes 是独立的一项。"""
    print("\n[2] 指纹基于全量清单，prefixes 单独成键")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        all_zips = _mk_zips(d, 5)
        for i in range(5, 9):                      # 再加 4 个 CX2，共 9 个
            (d / f"CX2_{i}_10_01_12.zip").write_bytes(b"PK\x03\x04fake")
        all_zips = sorted(d.glob("*.zip"))
        filtered = [z for z in all_zips if z.name.startswith("CS2_")]

        k_full = _calce_cache_key(d, all_zips, (2.85, 4.15), ["CS2"])
        k_filt = _calce_cache_key(d, filtered, (2.85, 4.15), ["CS2"])
        check("全量清单与过滤清单算出不同指纹（长度不同）",
              k_full["zip_count"] == 9 and k_filt["zip_count"] == 5)
        check("zip_count 记的是传入清单的长度", k_full["zip_count"] != k_filt["zip_count"])

        k_p1 = _calce_cache_key(d, all_zips, (2.85, 4.15), ["CS2"])
        k_p2 = _calce_cache_key(d, all_zips, (2.85, 4.15), ["CS2", "CX2"])
        k_p3 = _calce_cache_key(d, all_zips, (2.85, 4.15), None)
        check("prefixes 进指纹", k_p1["prefixes"] != k_p2["prefixes"])
        check("无 prefixes 与有 prefixes 不同", k_p3["prefixes"] == [])
        check("prefixes 之外的项不受 prefixes 影响",
              all(k_p1[k] == k_p2[k] for k in k_p1 if k != "prefixes"))


def test_valid_rejects_foreign_dir() -> None:
    """坑 4：换了目录，旧缓存绝不能被复用。"""
    print("\n[3] 换目录不得复用旧缓存")
    real_cache, real_meta = _CALCE_CACHE, _CALCE_CACHE_META
    try:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            zips = _mk_zips(d, 3)
            # 把缓存/指纹重定向到临时文件
            import data.loaders as L
            L._CALCE_CACHE = d / "cache.parquet"
            L._CALCE_CACHE_META = d / "cache.meta.json"

            # 没有缓存 -> 无效
            check("无缓存时无效", not _calce_cache_valid(d, zips, (2.85, 4.15), ["CS2"]))

            # 写一份"来自 A 目录"的指纹，然后问 B 目录是否有效
            key_a = _calce_cache_key(d / "A", zips, (2.85, 4.15), ["CS2"])
            L._CALCE_CACHE_META.write_text(json.dumps(key_a), encoding="utf-8")
            L._CALCE_CACHE.write_bytes(b"x")
            check("A 目录的指纹对 A 有效", _calce_cache_valid(d / "A", zips, (2.85, 4.15), ["CS2"]))
            check("A 目录的指纹对 B 无效",
                  not _calce_cache_valid(d / "B", zips, (2.85, 4.15), ["CS2"]))
            check("改了阈值也无效",
                  not _calce_cache_valid(d / "A", zips, (2.8, 4.1), ["CS2"]))
            check("改了 prefixes 也无效",
                  not _calce_cache_valid(d / "A", zips, (2.85, 4.15), ["CX2"]))
            check("zip 清单变了也无效",
                  not _calce_cache_valid(d / "A", zips[:2], (2.85, 4.15), ["CS2"]))
    finally:
        import data.loaders as L
        L._CALCE_CACHE, L._CALCE_CACHE_META = real_cache, real_meta


def test_load_calce_empty_dir() -> None:
    """坑 4 的另一面：空目录必须返回 0 支，不能拿旧缓存冒充。"""
    print("\n[4] 空目录返回 0 支（不返回缓存里的 14 支）")
    import contextlib
    import io

    real_cache, real_meta = _CALCE_CACHE, _CALCE_CACHE_META
    try:
        with tempfile.TemporaryDirectory() as td:
            import data.loaders as L
            L._CALCE_CACHE = Path(td) / "c.parquet"
            L._CALCE_CACHE_META = Path(td) / "c.meta.json"
            # 造一份"有 14 支"的假缓存，但指纹文件不存在 -> 必须判无效
            L._CALCE_CACHE.write_bytes(b"fake")
            with contextlib.redirect_stdout(io.StringIO()):
                out = load_calce(Path(td), ["CS2", "CX2"])
            check("空目录返回 0 支", len(out) == 0, f"实际 {len(out)} 支")
    finally:
        import data.loaders as L
        L._CALCE_CACHE, L._CALCE_CACHE_META = real_cache, real_meta


def test_no_duplicate_load_calce() -> None:
    """坑 1：文件里只能有一个 load_calce 定义。"""
    print("\n[5] load_calce 只定义一次")
    src = (ROOT / "data" / "loaders.py").read_text(encoding="utf-8")
    n = src.count("\ndef load_calce(")
    check(f"load_calce 定义数 == 1（实际 {n}）", n == 1)
    n2 = src.count("\ndef _calce_cache_key(")
    check(f"_calce_cache_key 定义数 == 1（实际 {n2}）", n2 == 1)


if __name__ == "__main__":
    print("=" * 62)
    print("CALCE 缓存指纹回归测试")
    print("=" * 62)
    test_key_is_path_form_independent()
    test_key_uses_full_zip_list()
    test_valid_rejects_foreign_dir()
    test_load_calce_empty_dir()
    test_no_duplicate_load_calce()
    print("\n" + "=" * 62)
    print(f"通过 {len(_PASS)} 项，失败 {len(_FAIL)} 项")
    if _FAIL:
        for f in _FAIL:
            print(f"  ✗ {f}")
        sys.exit(1)
    print("全部通过 ✓")
