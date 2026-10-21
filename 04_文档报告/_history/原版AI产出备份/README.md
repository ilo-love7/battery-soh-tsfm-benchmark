# 智诚时序基础模型零样本电池基准项目

## 📋 项目概述

本项目旨在利用先进的时序预测基础模型（TimesFM、Chromo、Llama-Moirai）进行电池寿命和容量的**零样本学习**研究，建立统一的评估基准。

### 核心目标
1. **零样本泛化**: 在未见过的电池数据上直接推理
2. **少样本微调**: 使用少量标注数据提升性能
3. **跨数据集迁移**: 验证模型从 MIT→NASA→CALCE 的迁移能力
4. **基准对比**: 系统评估不同基础模型的优劣

---

## 🏗️ 项目结构

```
智诚时序基础模型项目/
├── 01_源代码/
│   ├── data/                    # 数据处理模块
│   │   ├── loaders.py          # 数据集加载器
│   │   └── preprocessing.py    # 数据预处理工具
│   ├── models/                  # 模型实现
│   │   ├── timesfm_wrapper.py
│   │   ├── chromo_wrapper.py
│   │   └── moirai_wrapper.py
│   ├── training/                # 训练脚本
│   │   ├── train_zero_shot.py  # 零样本训练
│   │   ├── train_few_shot.py   # 少样本训练
│   │   └── config.yaml         # 配置文件
│   └── evaluation/              # 评估脚本
│       ├── metrics.py
│       └── ablation_study.py
├── 02_数据集链接/               # 数据集下载地址与说明
│   ├── mit_matr-1.md
│   ├── nasa_pcoe.md
│   └── calce_umd.md
├── 03_实验结果/                 # 实验输出
│   ├── figures/                # 图表输出
│   ├── tables/                 # 表格数据
│   └── logs/                   # 训练日志
├── 04_文档报告/                 # 文档材料
│   ├── 技术方案-v1.md
│   └── 实验报告-template.md
├── 05_环境配置/                 # 环境设置
│   ├── requirements.txt
│   ├── setup_env.bat           # Windows 环境脚本
│   └── .env.example            # 环境变量模板
├── notebooks/                   # Jupyter 探索性分析
├── scripts/                     # Shell 工具脚本
└── README.md                    # 本文件
```

---

## 🚀 快速开始

### 1. 环境设置

#### Windows:
```bash
cd "C:\Users\34648\Desktop\智诚时序基础模型项目"
05_环境配置\setup_env.bat
venv\Scripts\activate.bat
pip install -r 05_环境配置\requirements.txt
```

#### Linux/Mac:
```bash
conda create -n zeitung python=3.11
conda activate zeitung
pip install -r 05_环境配置/requirements.txt
```

### 2. 数据准备

将以下数据集复制到指定位置：
```
D:/项目/02_数据集/01_国际公开集/MIT_MATR-1/ → 02_数据集链接/MIT_DATA/
D:/项目/02_数据集/01_国际公开集/NASA_PCoE/ → 02_数据集链接/NASA_DATA/
D:/项目/02_数据集/01_国际公开集/CALCE_UMD/ → 02_数据集链接/CALCE_DATA/
```

### 3. 运行基准测试

```python
# 零样本推理 (MIT 数据集)
python 01_源代码/run_benchmark.py \
    --model chromo-bolt \
    --dataset mit \
    --setting zero-shot \
    --output-dir 03_实验结果/mit-zero-shot/

# 少样本微调
python 01_源代码/run_benchmark.py \
    --model timesfm \
    --dataset nasa \
    --setting few-shot \
    --n-support 10 \
    --output-dir 03_实验结果/nasa-few-shot/
```

---

## 📊 预期实验设计

### 实验组 W1 (基线)
- MIT 数据集基准性能
- NASA/CALCE 数据集 baseline

### 实验组 W2 (零样本)
- MIT-zero / NASA-zero / CALCE-zero
- Chromo/TimesFM/Moirai 直接推理

### 实验组 W3 (少样本)
- MIT-few / NASA-few / CALCE-few
- 10-50 samples 支持集

### 消融实验
- 移除注意力机制
- 归一化影响分析
- 上下文窗口大小敏感性

---

## 📈 性能指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| R² (MIT) | > 0.90 | 决定系数 |
| MAPE (平均) | < 5% | 平均绝对百分比误差 |
| 推理速度 | < 10ms/cell | 单电池预测耗时 |
| GPU 显存 | < 8GB | 训练/推理峰值占用 |

---

## 👥 团队信息

- **项目名称**: 智诚时序基础模型零样本电池基准
- **版本**: v1.0
- **创建日期**: 2026-10-19
- **团队**: 智诚 AI Research Team

---

## 📝 注意事项

1. ⚠️ **GPU 要求**: 建议 NVIDIA RTX A6000 或更高，当前 RTX 4050 可用于小规模实验
2. ⚠️ **内存需求**: ≥ 64GB RAM（推荐）
3. ⚠️ **存储需求**: ≥ 500GB SSD（用于数据集和缓存）
4. ✅ **先小规模验证**: 建议先用 MIT 子集（10 cells）验证流程
5. ✅ **数据隐私**: 注意知识产权和保密协议

---

## 🔗 相关链接

- [MIT Benchmark Protocol](https://data.matr.io/)
- [TimesFM GitHub](https://github.com/google-research/timesfm)
- [Chromo Forecasting](https://amazon.github.io/chromo-forcasting/)
- [Moirai UN2TS](https://github.com/salesforce/un2ts)

---

*最后更新：2026-10-19*