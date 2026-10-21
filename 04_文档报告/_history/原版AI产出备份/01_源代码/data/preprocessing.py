"""
MIT Battery Dataset Processing Script
版本：v1.0
作者：智诚 AI Team
日期：2026-10-19

功能：
- 加载 MIT/NASA/CALCE 电池数据集
- Savitzky-Golay 平滑滤波
- 容量衰减特征计算
- 数据增强（噪声注入）
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def load_mit_data(pkl_path: str) -> pd.DataFrame:
    """
    加载 MIT MATR-1 数据集
    
    Args:
        pkl_path: .pkl 文件路径
        
    Returns:
        DataFrame 包含 cycle, capacity, voltage 等列
    """
    try:
        import pickle
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # 转换为 DataFrame
        df = pd.DataFrame(data)
        return df
        
    except Exception as e:
        print(f"[ERROR] 加载 MIT 数据失败：{e}")
        return None


def load_nasa_data(folder_path: str, cell_id: int) -> pd.DataFrame:
    """
    加载 NASA PCoE 电池数据集
    
    Args:
        folder_path: 数据集文件夹路径
        cell_id: 电池编号 (如 5, 6, 7, 8)
        
    Returns:
        DataFrame 包含 Cycle, Voltage, Current, Capacity
    """
    file_path = f"{folder_path}/{cell_id}.csv"
    try:
        df = pd.read_csv(file_path, header=0)
        # 重命名列以统一格式
        df = df.rename(columns={
            'cycle': 'Cycle',
            'Voltage': 'Voltage',
            'Current': 'Current',
            'Capacity': 'Capacity'
        })
        return df
    except Exception as e:
        print(f"[ERROR] 加载 NASA 电池{cell_id}数据失败：{e}")
        return None


def load_calce_data(file_path: str) -> pd.DataFrame:
    """
    加载 CALCE UMD 电池数据集
    
    Args:
        file_path: ZIP 或 CSV 文件路径
        
    Returns:
        DataFrame
    """
    try:
        if file_path.endswith('.zip'):
            df = pd.read_csv(file_path, compression='zip')
        else:
            df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"[ERROR] 加载 CALCE 数据失败：{e}")
        return None


def apply_savitzky_golay(
    df: pd.DataFrame,
    column: str = 'capacity',
    window: int = 11,
    order: int = 3
) -> pd.DataFrame:
    """
    应用 Savitzky-Golay 平滑滤波
    
    Args:
        df: 输入数据框
        column: 需要平滑的列名
        window: 窗口大小（奇数）
        order: 多项式阶数
        
    Returns:
        添加 smoothed 列的数据框
    """
    df[f'{column}_smoothed'] = savgol_filter(
        df[column].values,
        window_length=window,
        polyorder=order
    )
    return df


def calculate_decay_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    计算电池退化指标
    
    Args:
        df: 包含 capacity 和 cycle 列的数据框
        
    Returns:
        字典包含 cycle_life, capacity_fade_rate, retention_at_end
    """
    initial_capacity = df['capacity'].iloc[0]
    final_capacity = df['capacity'].iloc[-1]
    cycle_life = df['cycle'].max()
    
    metrics = {
        'cycle_life': float(cycle_life),
        'capacity_fade_rate': float((initial_capacity - final_capacity) / cycle_life),
        'retention_at_end': float(final_capacity / initial_capacity * 100),
        'initial_capacity': float(initial_capacity)
    }
    
    return metrics


def augment_data_with_noise(
    series: np.ndarray,
    noise_std: float = 0.01,
    seed: int = 42
) -> np.ndarray:
    """
    数据增强：添加高斯噪声
    
    Args:
        series: 原始时间序列
        noise_std: 噪声标准差（相对于原始 std）
        seed: 随机种子
        
    Returns:
        带噪声的时间序列
    """
    np.random.seed(seed)
    noise = np.random.normal(0, noise_std * np.std(series), size=series.shape)
    augmented = series + noise
    return augmented


def normalize_series(
    series: np.ndarray,
    method: str = 'zscore'
) -> Tuple[np.ndarray, float, float]:
    """
    时间序列标准化
    
    Args:
        series: 输入序列
        method: 标准化方法 ('zscore' 或 'minmax')
        
    Returns:
        (标准化后的序列，均值，标准差)
    """
    if method == 'zscore':
        mean = np.mean(series)
        std = np.std(series)
        normalized = (series - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = np.min(series)
        max_val = np.max(series)
        normalized = (series - min_val) / (max_val - min_val + 1e-8)
    else:
        raise ValueError(f"未知标准化方法：{method}")
    
    return normalized, mean, std


def train_test_split_time_series(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    horizon: int = 100
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    时间序列划分训练集和测试集
    
    Args:
        df: 完整数据框
        train_ratio: 训练集比例
        horizon: 测试集步长
        
    Returns:
        (train_df, test_df)
    """
    n_samples = len(df)
    train_size = int(n_samples * train_ratio)
    
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    return train_df, test_df


class BatteryDataPreprocessor:
    """电池数据预处理器类"""
    
    def __init__(
        self,
        dataset_type: str = 'mit',
        smoothing_window: int = 11,
        smoothing_order: int = 3
    ):
        """
        初始化预处理器
        
        Args:
            dataset_type: 数据集类型 ('mit', 'nasa', 'calce')
            smoothing_window: Savitzky-Golay 窗口大小
            smoothing_order: Savitzky-Golay 多项式阶数
        """
        self.dataset_type = dataset_type
        self.smoothing_window = smoothing_window
        self.smoothing_order = smoothing_order
        
    def preprocess(self, raw_data, **kwargs) -> Dict[str, any]:
        """
        统一预处理接口
        
        Args:
            raw_data: 原始数据（DataFrame 或文件路径）
            **kwargs: 其他参数
            
        Returns:
            包含处理结果和元信息的字典
        """
        results = {
            'dataset_type': self.dataset_type,
            'metrics': {},
            'data': None,
            'metadata': {}
        }
        
        if isinstance(raw_data, str):
            # 从文件加载
            if self.dataset_type == 'mit':
                df = load_mit_data(raw_data)
            elif self.dataset_type == 'nasa':
                # 需要从 kwargs 获取 cell_id
                cell_id = kwargs.get('cell_id', 5)
                df = load_nasa_data(raw_data.replace('/DATA/', ''), cell_id)
            elif self.dataset_type == 'calce':
                df = load_calce_data(raw_data)
            else:
                raise ValueError(f"不支持的数据集类型：{self.dataset_type}")
        else:
            df = raw_data
        
        if df is not None:
            # 平滑处理
            df = apply_savitzky_golay(
                df, 
                column='capacity',
                window=self.smoothing_window,
                order=self.smoothing_order
            )
            
            # 计算指标
            results['metrics'] = calculate_decay_metrics(df)
            
            # 划分训练测试集
            train_df, test_df = train_test_split_time_series(df)
            results['data'] = {
                'train': train_df,
                'test': test_df
            }
            
            results['metadata'] = {
                'n_cells': len(df['cycle'].unique()) if 'cycle' in df.columns else 1,
                'n_cycles': len(df),
                'sample_rate': kwargs.get('sample_rate', 'unknown')
            }
        
        return results


# 示例用法
if __name__ == "__main__":
    print("=" * 50)
    print("电池数据预处理工具 - 测试运行")
    print("=" * 50)
    
    # 示例：检查工具是否可用
    print("\n[OK] 所有依赖包已导入")
    print("- numpy:", np.__version__)
    print("- pandas:", pd.__version__)
    print("- scipy:", scipy.__version__)
    
    print("\n[INFO] 请配置数据路径后运行实际处理")