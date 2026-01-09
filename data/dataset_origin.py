import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os
from pathlib import Path
import pandas as pd
import re

class H5StockDataset(Dataset):
    def __init__(self, h5_dir, label_dir, start_date, end_date):
        self.h5_dir = Path(h5_dir)
        self.label_dir = label_dir
        self.start_date = start_date
        self.end_date = end_date
        self.labels_df = self._load_labels()
        self.h5_files = self._get_h5_files()
        self.data_cache = {}  # 存储数据: {(date, stock_code): features}
        self.sample_infos = self._load_all_data()
        print(f"数据集加载完成: {len(self.h5_files)}个文件, {len(self.sample_infos)}个样本")

    def _get_h5_files(self):
        """获取H5文件列表"""
        all_h5 = sorted(self.h5_dir.glob('*.h5'))
        h5_files = []
        for h5_file in all_h5:
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', h5_file.name)
            # 检查文件名是否包含日期
            if date_match:
                file_date = date_match.group(1)
                if self.start_date <= file_date <= self.end_date:
                    h5_files.append(h5_file)

        return h5_files

    def _load_labels(self):
        """加载标签CSV文件"""
        # 读取CSV
        df = pd.read_csv(self.label_dir)
        df = df.dropna(subset=['label'])
        # 转换label为整数
        df['label'] = df['label'].astype(int)
        df['datetime'] = df['datetime'].astype(str).str.replace('-', '')
        df['stock_code'] = df['stock_code'].str.replace('.SZ', '').str.replace('.SH', '')

        return df

    def _load_all_data(self):
        h5_records = [ ]# 先构建h5列表映射
        h5_file_map = {}  
        for h5_file in self.h5_files:
            match = re.search(r'(\d{4}-\d{2}-\d{2})', h5_file.name)
            file_date = match.group(1).replace('-', '')  # 20170103
            h5_file_map[file_date] = h5_file
            
            with h5py.File(h5_file, 'r') as f:
                # 不具体加载数据
                for stock_code in f.keys():
                    h5_records.append({
                        'datetime': file_date,
                        'stock_code': stock_code
                    })
        
        h5_df = pd.DataFrame(h5_records)
        # merge匹配，标签df作为主表
        merged_df = pd.merge(
            self.labels_df[['datetime', 'stock_code', 'label']],
            h5_df,
            on=['datetime', 'stock_code'],
            how='inner'
        )
        merged_by_date = merged_df.groupby('datetime')
        samples = []
        for file_date, day_samples in merged_by_date:
            h5_file = h5_file_map[file_date]
            
            with h5py.File(h5_file, 'r') as f:
                for _, row in day_samples.iterrows():
                    stock_code = row['stock_code']
                    # 只加载merge后匹配成功的数据
                    features = f[stock_code][:]
                    cache_key = (file_date, stock_code)
                    self.data_cache[cache_key] = features
                    samples.append({
                        'date': file_date,
                        'stock_code': stock_code,
                        'label': row['label']
                    })
        
        return samples
        
    def __len__(self):
        return len(self.sample_infos)
    
    def __getitem__(self, idx):
        """从内存中获取特征和标签"""
        sample = self.sample_infos[idx]
        cache_key = (sample['date'], sample['stock_code'])
        features = torch.from_numpy(self.data_cache[cache_key]).float()
        
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return {
            'features': features,  # (7, 8, 8)
            'label': label,        # 0, 1, 2
            'stock_code': sample['stock_code'],
            'date': sample['date']
        }