# 数据集说明（零复制）

本目录**只放说明，不放数据**。三个数据集都不随仓库分发，请按下文各文档里的官方链接自行获取，
然后让代码指向它——两种方式任选：

```bash
# 方式 A：环境变量（推荐，不动代码）
export BATTERY_DATA_ROOT=/path/to/datasets
export MIT_CELLS=/path/to/mit_cells.parquet      # 也可逐个覆盖
export NASA_MAT_DIR=/path/to/1.\ BatteryAgingARC-FY08Q4
export CALCE_DIR=/path/to/CALCE_UMD
```

```python
# 方式 B：复制 01_源代码/configs/local_paths.example.py 为 configs/local_paths.py 后填写
DATASET_OVERRIDES = {"calce": {"dir": "/path/to/CALCE_UMD"}}
```

默认布局是 `<项目根>/data/{MIT,NASA_PCoE,CALCE_UMD}`，按这个摆也可零配置。

| 数据集 | 需要的文件 | 格式 | 规模 |
|---|---|---|---|
| MIT-Stanford (MATR-1) | `mit_cells.parquet` + `mit_cycles.parquet` | 长表 parquet（由 `MATR.zip` 清洗得到） | 124 cells：train 41（batch1）/ test 43（batch2）/ secondary_test 40（batch3）；100 501 条循环记录 |
| MIT 原始包 | `MATR.zip` | 每 cell 一个 pkl（嵌套 dict，含逐循环 V/I/T） | 4.86 GB，仅用于重新生成上面的 parquet |
| NASA PCoE | `1. BatteryAgingARC-FY08Q4/*.mat` | MATLAB struct | 4 cells：B0005 / B0006 / B0007 各 168 循环，B0018 132 循环 |
| CALCE UMD | `CS2_*.zip` / `CX2_*.zip` | zip 内 xlsx | 27 个 zip → 按电压窗口筛出 **14 支全深度老化**（见下） |

> CALCE 的 14 与常见的"23 cells"不同：本基准只保留**真正从满充放到截止电压**的 cell，
> 把中途断测的排除在外（否则容量序列会出现伪"跳水"）。完整的筛选链（27 → 22 → 14）
> 与 13 支被剔除的逐条理由见 [`CALCE_UMD.md`](CALCE_UMD.md)。

## 读取方式

```python
import sys; sys.path.insert(0, "01_源代码")
from configs.config import DATASETS, check_dataset
from data.loaders import load_dataset

check_dataset("mit")                                   # 路径不对时给出可读报错
cells = load_dataset("mit", DATASETS, split="test")    # 43 cells
cells = load_dataset("nasa", DATASETS)                 # 4 cells
cells = load_dataset("calce", DATASETS)                # 14 cells
# 每个 cell -> DataFrame(cycle, capacity_Ah)
```

CALCE 首次解析约 10–20 分钟，之后走 `01_源代码/data/processed/calce_capacity_v3.parquet` 缓存，秒开。

## 各数据集文档

- [`MIT_MATR-1.md`](MIT_MATR-1.md) —— 来源、字段、EOL 标签说明、从原始 zip 生成 parquet 的脚本
- [`NASA_PCoE.md`](NASA_PCoE.md) —— .mat 结构与解析要点
- [`CALCE_UMD.md`](CALCE_UMD.md) —— zip/xlsx 结构、全深度筛选、断测段处理
