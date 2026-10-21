# NASA PCoE 电池数据集说明

## 📊 数据集概览

| 属性 | 值 |
|------|-----|
| **名称** | NASA Prognostics Center of Excellence (PCoE) |
| **发布机构** | NASA Ames Research Center |
| **化学体系** | Li-ion 多元正极 |
| **电池数量** | 5 只（编号 5, 6, 7, 8, 9） |
| **实验条件** | 多种充放电策略组合 |
| **参考论文** | [Hedman et al., AIAA SciTech 2016](https://doi.org/10.2514/6.2016-1335) |

---

## 💾 本地存储位置

```
D:/项目/02_数据集/01_国际公开集/NASA_PCoE/
├── 1. BatteryAgingARC-FY08Q4/       # FY08 Q4 数据
├── 2. BatteryAgingARC_25_26_27_28_P1/  # P1 数据集
└── ... (多个压缩包)
```

---

## 🔧 数据加载示例

```python
import pandas as pd

# 加载单个电池数据
cell_id = 5
file_path = f'D:/项目/02_数据集/01_国际公开集/NASA_PCoE/{cell_id}.csv'
df = pd.read_csv(file_path, header=0)

print(df.columns)
# ['Cycle', 'Voltage', 'Current', 'Temperature', 'Capacity']
```

---

## ⚙️ 预处理步骤

```python
from scipy.signal import savgol_filter

# 平滑容量曲线
window = 11
order = 3
df['capacity_smoothed'] = savgol_filter(
    df['Capacity'].values,
    window_length=window,
    polyorder=order
)
```

---

## 🎯 使用建议

- **零样本测试**: 用 4 只电池训练，第 5 只测试
- **少样本微调**: 每只电池提供 10 个循环作为支持集
- **Horizon 设置**: 预测未来 100-1000 个循环

---

*文档版本：v1.0 | 最后更新：2026-10-19*