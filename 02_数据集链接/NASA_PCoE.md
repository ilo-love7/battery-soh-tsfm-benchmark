# NASA PCoE 电池数据集（真实结构）

| 项 | 内容 |
|---|---|
| 名称 | NASA Prognostics Center of Excellence (PCoE) Battery Aging ARC |
| 本地路径 | `$NASA_MAT_DIR`（默认 `<项目根>/data/NASA_PCoE/1. BatteryAgingARC-FY08Q4/`） |
| 格式 | `.mat`（MATLAB 结构体），**不是 csv** |
| 本库可用电池 | **4 只：B0005 / B0006 / B0007 / B0018** |
| 循环数 | B0005/6/7 各 168 个放电循环；B0018 为 132 个 |
| 出处 | https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ |

> ⚠️ 原文档写"5 只（编号 5,6,7,8,9）"并给出 `pd.read_csv('5.csv')` 示例——两者都不对。
> 本库 FY08Q4 包内只有 B0005/6/7/18 四只，且为 .mat 格式。

## 结构

```python
import scipy.io as sio
m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
struct = m["B0005"]              # 顶层字段只有 cycle
for c in struct.cycle:           # 616 个循环结构（含充/放/阻抗）
    if c.type == "discharge":
        cap = float(c.data.Capacity)   # 该循环放电容量（Ah）
```

- `cycle[i].type` ∈ {charge, discharge, impedance}
- 只有 discharge 循环有 `Capacity`；charge 循环给的是电压/电流/温度曲线
- 中途插入阻抗测试会导致循环号不连续，属正常

## 预处理

- 只取 discharge 循环，按时间顺序编号为 cycle 1..N；
- 平滑参考线 Savitzky-Golay(window=11, order=2)，仅用于评估；
- 本项目由 `01_源代码/data/loaders.py::load_nasa` 统一处理，不要另写一套。

## 用途

跨库零样本测试（**不参与任何训练**）。4 只电池样本太少，不做 train/test 划分。
未衰减到 80% 的电池不参与 EOL 统计（见方案 v2 §3.3）。
