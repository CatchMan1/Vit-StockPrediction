import torch
from torch.utils.data import Dataset
import numpy as np


class MockLevel2Dataset(Dataset):
    """
    模拟 Level-2 数据集，用于测试训练流程
    生成形状为 (20, 15, 8, 8) 的特征张量和 (1,) 的回归标签
    """
    def __init__(self, num_samples=1000, seq_len=20, channels=15, height=8, width=8, seed=42):
        """
        Args:
            num_samples: 样本数量
            seq_len: 时间序列长度（ViViT 使用）
            channels: 通道数（15个特征通道）
            height: 图像高度
            width: 图像宽度
            seed: 随机种子
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.channels = channels
        self.height = height
        self.width = width
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        返回:
            features: (20, 15, 8, 8) 的特征张量
            label: (1,) 的回归标签（收益率）
        """
        np.random.seed(idx)
        
        features = torch.randn(self.seq_len, self.channels, self.height, self.width)
        
        features = torch.clamp(features, -3, 3)
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        label = torch.randn(1) * 0.05
        
        return features, label


class MockLevel2DatasetViT(Dataset):
    """
    模拟 Level-2 数据集（ViT 版本）
    生成形状为 (15, 8, 8) 的特征张量和 (1,) 的回归标签
    """
    def __init__(self, num_samples=1000, channels=15, height=8, width=8, seed=42):
        """
        Args:
            num_samples: 样本数量
            channels: 通道数（15个特征通道）
            height: 图像高度
            width: 图像宽度
            seed: 随机种子
        """
        self.num_samples = num_samples
        self.channels = channels
        self.height = height
        self.width = width
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        返回:
            features: (15, 8, 8) 的特征张量
            label: (1,) 的回归标签（收益率）
        """
        np.random.seed(idx)
        
        features = torch.randn(self.channels, self.height, self.width)
        
        features = torch.clamp(features, -3, 3)
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        label = torch.randn(1) * 0.05
        
        return features, label
