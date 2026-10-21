"""本机路径覆盖（示例）。

复制为同目录下的 ``local_paths.py`` 后按实际情况修改；该文件已被 .gitignore 排除，
不会被提交。不改也完全可以——只要按 config.py 顶部的默认布局放数据，
或者用环境变量（MIT_CELLS / NASA_MAT_DIR / CALCE_DIR / BATTERY_DATA_ROOT）指定。
"""

DATASET_OVERRIDES = {
    "mit": {
        "cells": "D:/datasets/MIT/mit_cells.parquet",
        "cycles": "D:/datasets/MIT/mit_cycles.parquet",
        "matr_zip": "D:/datasets/MIT/MATR.zip",
    },
    "nasa": {
        "dir": "D:/datasets/NASA_PCoE",
        "mat_dir": "D:/datasets/NASA_PCoE/1. BatteryAgingARC-FY08Q4",
    },
    "calce": {
        "dir": "D:/datasets/CALCE_UMD",
    },
}
