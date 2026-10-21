"""统一路径与协议配置。

所有路径集中在此，代码里不硬编码。数据只读取，不复制、不改写原始文件。

数据路径的解析顺序（先命中者生效）：
  1. 数据集专属环境变量，如 ``MIT_CELLS`` / ``NASA_MAT_DIR`` / ``CALCE_DIR``；
  2. 总根环境变量 ``BATTERY_DATA_ROOT`` + 约定的子目录布局；
  3. 项目内默认 ``<项目根>/data`` + 同样的子目录布局。

默认布局（把数据放成这样即可零配置运行）::

    data/
    ├── MIT/
    │   ├── mit_cells.parquet
    │   ├── mit_cycles.parquet
    │   └── MATR.zip
    ├── NASA_PCoE/1. BatteryAgingARC-FY08Q4/
    └── CALCE_UMD/            # 内含 CS2_* / CX2_* 的 xlsx
"""
import os
from pathlib import Path

# ---- 项目根 ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---- 网络：部分网络下 huggingface.co 不通，权重下载统一走镜像 ----
# 外部若已设置 HF_ENDPOINT 则尊重外部值（海外环境可设为 https://huggingface.co）。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# 统一缓存到项目内，避免散落 C 盘各处；体积可控（small 版权重 < 1 GB）
_HF_HOME = PROJECT_ROOT / "shared" / "hf_cache"
_HF_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_HOME))


def _env_path(var: str, default: Path) -> Path:
    """环境变量优先；为空则用默认值。支持 `~` 展开。"""
    raw = os.environ.get(var)
    return Path(raw).expanduser() if raw else default


# ---- 数据根 ----
DATA_ROOT = _env_path("BATTERY_DATA_ROOT", PROJECT_ROOT / "data")

# ---- 数据路径（每个都可用环境变量单独覆盖） ----
# MIT-Stanford：需要 cells/cycles 两张长表（由 MATR.zip 清洗得到），
#   没有现成 parquet 时可用 02_数据集链接/MIT_Stanford.md 里的脚本从原始 zip 生成。
# NASA PCoE：需要 .mat 文件目录（B0005/B0006/B0007/B0018）。
# CALCE UMD：需要 CS2_*/CX2_* 的 xlsx 目录。
DATASETS = {
    "mit": {
        "cells": _env_path("MIT_CELLS", DATA_ROOT / "MIT" / "mit_cells.parquet"),
        "cycles": _env_path("MIT_CYCLES", DATA_ROOT / "MIT" / "mit_cycles.parquet"),
        "matr_zip": _env_path("MIT_MATR_ZIP", DATA_ROOT / "MIT" / "MATR.zip"),
    },
    "nasa": {
        "dir": _env_path("NASA_DIR", DATA_ROOT / "NASA_PCoE"),
        "mat_dir": _env_path("NASA_MAT_DIR",
                             DATA_ROOT / "NASA_PCoE" / "1. BatteryAgingARC-FY08Q4"),
        "cells": ["B0005", "B0006", "B0007", "B0018"],
    },
    "calce": {
        "dir": _env_path("CALCE_DIR", DATA_ROOT / "CALCE_UMD"),
        # 常用系列：CS2（1.1 Ah LCO/石墨）与 CX2（1.35 Ah LCO/石墨）
        "prefixes": ["CS2", "CX2"],
    },
}

# ---- 可选的本机覆盖：configs/local_paths.py（已被 .gitignore 排除） ----
# 内容形如：
#   DATASET_OVERRIDES = {"mit": {"cells": "D:/datasets/mit/mit_cells.parquet"}, ...}
# 优先级最高，方便"代码在 C 盘、数据在 D 盘"的本地开发；对外发布时不带这个文件。
try:  # pragma: no cover - 纯本地便利机制
    from configs.local_paths import DATASET_OVERRIDES  # type: ignore
except Exception:  # ImportError 或文件不存在
    DATASET_OVERRIDES = {}
for _ds, _kv in (DATASET_OVERRIDES or {}).items():
    DATASETS.setdefault(_ds, {}).update(
        {_k: Path(_v).expanduser() for _k, _v in _kv.items()}
    )

# 每个数据集"必须存在"的键，用于启动时给出可读报错而不是 KeyError
_REQUIRED = {"mit": ["cells"], "nasa": ["mat_dir"], "calce": ["dir"]}

# 缺失时提示用户设置哪个环境变量
_ENV_HINT = {
    ("mit", "cells"): "MIT_CELLS",
    ("mit", "cycles"): "MIT_CYCLES",
    ("mit", "matr_zip"): "MIT_MATR_ZIP",
    ("nasa", "dir"): "NASA_DIR",
    ("nasa", "mat_dir"): "NASA_MAT_DIR",
    ("calce", "dir"): "CALCE_DIR",
}


def check_dataset(dataset: str) -> None:
    """确认该数据集的必需路径存在；不存在就抛出带修复提示的错误。"""
    if dataset not in DATASETS:
        raise SystemExit(f"未知数据集 {dataset!r}，可选：{list(DATASETS)}")
    for key in _REQUIRED[dataset]:
        p = Path(DATASETS[dataset][key])
        if not p.exists():
            var = _ENV_HINT.get((dataset, key), key.upper())
            raise SystemExit(
                f"找不到 {dataset} 的数据：{p}\n"
                f"请把数据放到上述位置，或设置环境变量 {var} 指向真实路径。\n"
                f"数据集获取方式见 02_数据集链接/{dataset.upper()}*.md"
            )


# ---- MIT 官方三划分（41/43/40） ----
MIT_SPLITS = {
    "train": "batch1",            # 41 cells，few-shot 微调用
    "test": "batch2",             # 43 cells，主测试集
    "secondary_test": "batch3",   # 40 cells，MIT 内部 OOD
}

# ---- 基础模型注册表（真实 HF 仓库名） ----
MODELS = {
    # 主用：Bolt-small 190.9 MB，CPU 快，直接输出分位数
    "chronos-bolt": {
        "adapter": "chronos", "kwargs": {"bolt": True, "model_id": "amazon/chronos-bolt-small"},
        "note": "Amazon Chronos-Bolt small（190.9 MB, apache-2.0）",
    },
    # 对照：T5 采样式，同尺寸，用来验证 Bolt 的 patch 结构是否影响结论
    "chronos": {
        "adapter": "chronos", "kwargs": {"bolt": False, "model_id": "amazon/chronos-t5-small"},
        "note": "Amazon Chronos T5-small（采样式，取样本中位数）",
    },
    "timesfm": {
        "adapter": "timesfm", "kwargs": {},
        "note": "Google TimesFM 2.5 200M（需独立环境，torch 后端）",
    },
    "moirai": {
        "adapter": "moirai", "kwargs": {},
        "note": "Salesforce Moirai-1.0-R-small（uni2ts + gluonts）",
    },
    "lag-llama": {
        "adapter": "lagllama", "kwargs": {},
        "note": "社区 Lag-Llama（gluonts，独立环境）",
    },
}

# ---- 评测协议（写死，保证所有模型同口径） ----
PROTOCOL = {
    "start_frac": 0.5,        # 预测起点：序列长度 × 0.5
    "start_frac_sensitivity": [0.4, 0.5, 0.6],  # 敏感性分析
    "smooth_window": 11,      # Savitzky-Golay 平滑（仅用于评估参考）
    "smooth_order": 2,
    "outlier_thresh": 0.03,   # 相对局部中位数偏离阈值
    "soh_ref_points": 5,      # SOH 归一化参考：上下文前 5 点中位数
    "eol_threshold": 80.0,    # EOL = SOH 首次 ≤ 80%
    "rolling_chunk": 64,      # 递归预测单次 horizon
    "seeds": [42, 43, 44],    # 采样类模型重复种子
}

# ---- 输出 ----
RESULTS_ROOT = PROJECT_ROOT / "03_实验结果"

# ---- 权重：优先用项目内已下载的本地副本，避免每次联网 + 规避 hf-xet 中断 ----
MODELS_DIR = PROJECT_ROOT / "shared" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def resolve_model_ref(repo_id: str) -> str:
    """把 HF 仓库名解析成可加载路径：本地副本优先，否则回落仓库名（走镜像）。

    本地副本目录名 = 仓库名最后一段，如 amazon/chronos-bolt-small
    -> shared/models/chronos-bolt-small/
    """
    local = MODELS_DIR / repo_id.rstrip("/").split("/")[-1]
    if (local / "config.json").exists():
        return str(local)
    return repo_id
