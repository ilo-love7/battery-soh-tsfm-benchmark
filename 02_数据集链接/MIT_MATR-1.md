# MIT MATR-1 电池数据集说明

## 📊 数据集概览

| 属性 | 值 |
|------|-----|
| **名称** | MIT-Stanford MATR-1 (Severson 2019) |
| **发布机构** | MIT / Stanford / Toyota Research Institute |
| **化学体系** | LCO/石墨 商用软包 |
| **电池数量** | 124 只 |
| **实验策略** | 72 种不同的快充策略 |
| **采样频率** | 100s (约 1.67 分钟/循环) |
| **关键指标** | cycle_life, capacity, voltage, current |
| **参考论文** | [Severson et al., Nature Energy 2019](https://doi.org/10.1038/s41560-019-0356-8) |

---

## 💾 本地存储位置

```
$MIT_MATR_ZIP 所在目录（默认 <项目根>/data/MIT/）
├── MATR.zip                          # 统一格式 .pkl (4864 MB)
└── 原始数据/
    ├── 2017-05-12_batchdata.mat      # MATLAB 原始格式 (3 GB)
    ├── 2017-06-30_batchdata.mat      # MATLAB 原始格式 (2 GB)
    └── 2018-04-12_batchdata.mat      # MATLAB 原始格式 (3 GB)
```

---

## 🔧 数据加载方式

### 方法 1: 使用统一格式 (.pkl)

```python
import pickle
import pandas as pd

with open(os.environ.get('MIT_MATR_ZIP', 'data/MIT/MATR.zip'), 'rb') as f:
    data = pickle.load(f)

df = pd.DataFrame(data)
print(df.head())
```

### 方法 2: 使用 MATLAB 原始数据

```python
import scipy.io

mat_file = scipy.io.loadmat(
    'data/MIT/原始数据/'
    '2017-05-12_batchdata_updated_struct_errorcorrect.mat'
)

# 访问数据结构
print(mat_file.keys())
```

---

## 📈 数据内容

### 字段说明

| 字段名 | 类型 | 含义 | 单位 |
|--------|------|------|------|
| `cycle` | int | 循环编号 | - |
| `capacity` | float | 放电容量 | mAh |
| `cycle_life` | int | 总循环寿命 | - |
| `voltage` | list | 电压曲线 | V |
| `current` | list | 电流曲线 | A |
| `time` | float | 时间点 | s |

### 统计信息

```python
# 示例统计数据
- 初始平均容量：~2000 mAh
- 平均循环寿命：7.8 小时（约 28 天连续测试）
- 容量衰减曲线：Savitzky-Golay 平滑窗口=11, order=3
```

---

## ⚙️ 预处理建议

### Savitzky-Golay 平滑处理

```python
from scipy.signal import savgol_filter

window = 11   # 窗口大小
order = 3     # 多项式阶数

# 对容量数据进行平滑
capacity_smoothed = savgol_filter(
    df['capacity'].values,
    window_length=window,
    polyorder=order
)
```

### 数据增强（可选）

```python
# 添加高斯噪声用于鲁棒性训练
noise_std = 0.01 * np.std(df['capacity'])
augmented_capacity = df['capacity'] + np.random.normal(0, noise_std)
```

---

## 🎯 在基准测试中的使用

### 零样本推理 (Zero-shot)

```bash
python run_benchmark.py \
    --model chromo-bolt \
    --dataset mit \
    --setting zero-shot \
    --output-dir results/mit-zero-shot/
```

### 少样本微调 (Few-shot)

```bash
python run_benchmark.py \
    --model timesfm \
    --dataset mit \
    --setting few-shot \
    --n-support 10 \
    --output-dir results/mit-few-shot/
```

---

## 📋 数据划分建议

| 划分类型 | 训练集 | 测试集 | 用途 |
|----------|--------|--------|------|
| **跨电池泛化** | 90 cells | 34 cells | 未见过的电池 |
| **时间序列预测** | 前 80% cycles | 后 20% cycles | 未来循环预测 |
| **消融实验** | 全部数据 | 留出子集 | 组件贡献分析 |

---

## ⚠️ 注意事项

1. **内存需求**: 完整数据集需 ~15GB RAM（包括原始和统一格式）
2. **磁盘空间**: 建议预留 20GB 用于缓存和中间文件
3. **数据隐私**: 部分数据可能受 NDA 限制，请注意保密协议
4. **引用规范**: 
   ```bibtex
   @article{severson2019automated,
       title={Automated prediction of lithium-ion battery lifetime},
       author={Severson, Kevin A and others},
       journal={Nature Energy},
       volume={4},
       number={6},
       pages={385--392},
       year={2019}
   }
   ```

---

## 🔗 相关资源

- **官方下载**: https://data.matr.io/1
- **论文链接**: https://www.nature.com/articles/s41560-019-0356-8
- **GitHub 复现代码**: 搜索 "Severson 2019 battery lifetime"

---

*文档版本：v1.0*  
*最后更新：2026-10-19*  
*维护团队：智诚 AI Research Team*