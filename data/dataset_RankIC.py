import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os
from pathlib import Path
import pandas as pd

class H5StockDataset(Dataset):
    """
    从 H5 文件读取股票数据的 Dataset
    每个 H5 文件对应一天的数据，包含多只股票
    每只股票的数据形状为 (7, 8, 8)
    """
    def __init__(self, h5_dir, start_date='2017-01-03', end_date='2017-01-04'):
        self.h5_dir = Path(h5_dir)
        
        # 获取所有文件及其日期
        all_files = []
        for h5_file in sorted(self.h5_dir.glob('*.h5')):
            # 提取日期
            date_part = h5_file.stem[-10:]  # 取最后10个字符：YYYY-MM-DD
            if re.match(r'\d{4}-\d{2}-\d{2}', date_part):
                all_files.append((date_part, h5_file))
        
        # 日期切片
        self.h5_files = [
            h5_file for date_part, h5_file in all_files 
            if start_date <= date_part <= end_date
        ]
        self.date_groups = self._create_date_groups()
        self.samples = self._create_samples()


    def _create_date_groups(self):
        """创建按日期分组的截面数据"""
        date_groups = {}
        
        for h5_file in self.h5_files:
            # 提取日期
            date_part = h5_file.stem[-10:]  # YYYY-MM-DD
            date_key = date_part.replace('-', '')  # 20170103
            
            # 读取H5文件
            with h5py.File(h5_file, 'r') as f:
                # 获取所有股票代码
                stock_codes = list(f.keys())
                # 为每只股票创建基本信息
                date_groups[date_key] = []
                for stock_code in stock_codes:
                    date_groups[date_key].append({
                        'stock_code': stock_code,
                        'date': date_key,
                        'file_path': str(h5_file)
                    })
    
        return date_groups
    def _create_samples(self):
        """创建样本列表"""
        samples = []
        for date_key, stock_list in self.date_groups.items():
            for stock_info in stock_list:
                samples.append(
                    {
                    'date': date_key,
                    'stock_code': stock_info['stock_code'],
                    'file_path': stock_info['file_path']
                }
                )
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        
        sample = self.samples[idx]
        with h5py.File(sample['file_path'], 'r') as f:
            # 读取样本股票的数据
            data_array = f[sample["stock_code"]][:]
            
        # 转换为torch张量
        features = torch.from_numpy(data_array).float()
        
        return {
            'features': features,  # torch.Tensor, 形状 (7, 8, 8)
            'stock_code': sample['stock_code'],
            'date': sample['date']
        }
    
    
