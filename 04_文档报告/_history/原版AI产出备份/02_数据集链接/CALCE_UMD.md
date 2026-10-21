# CALCE UMD 电池数据集说明

## 📊 数据集概览

| 属性 | 值 |
|------|-----|
| **名称** | Center for Advanced Life Cycle Engineering (CALCE) |
| **发布机构** | University of Maryland (UMD) |
| **化学体系** | 多种 LCO, NMC, LFP |
| **电池数量** | 10+ 只（包含 SP1/SP2/SP3 等） |
| **测试类型** | DST, FUDS, US06, BJDST 等工况 |
| **参考链接** | [CALCE 官网](https://calce.umd.edu/battery-data-set) |

---

## 💾 本地存储位置

```
D:/项目/02_数据集/01_国际公开集/CALCE_UMD/
├── A123_DST-US06-FUDS-*.zip      # A123 系列电池
├── CS2_*.zip                      # CS2 系列电池
├── CX2_*.zip                      # CX2 系列电池
└── MANIFEST.csv                   # 数据清单
```

---

## 🔧 数据加载方式

```python
import pandas as pd

# 加载 CSV 格式
df = pd.read_csv('D:/项目/02_数据集/01_国际公开集/CALCE_UMD/MANIFEST.csv')

# 或解压 ZIP 文件
df_zip = pd.read_csv(
    'D:/项目/02_数据集/01_国际公开集/CALCE_UMD/CS2_3.zip',
    compression='zip'
)
```

---

## ⚙️ 预处理建议

```python
# 处理多温度条件下的数据
for temp in [-40, 0, 25, 40, 50]:
    file_path = f'Capacity_{temp}C.zip'
    df_temp = load_calce_data(file_path)
    
    # 温度补偿或单独建模
    apply_temperature_compensation(df_temp)
```

---

## 🎯 使用场景

- **零样本迁移**: MIT → CALCE 跨数据集泛化
- **工况鲁棒性**: 不同放电策略下的性能评估
- **不确定性量化**: 预测区间估计

---

*文档版本：v1.0 | 最后更新：2026-10-19*