import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os
from pathlib import Path
import pandas as pd
import re
from joblib import Parallel, delayed
from tqdm import tqdm  # 进度条库

class H5StockDataset(Dataset):
    def __init__(self, h5_dir, label_dir, start_date, end_date, num_workers):
        self.h5_dir = Path(h5_dir)
        self.label_dir = label_dir
        self.start_date = start_date
        self.end_date = end_date
        self.num_workers = num_workers
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

    # 多进程无法共享内存空间，因此还需要保存data_cache
    def _eval_one_task(self, task_params):
        data_cache = {}
        sample = []
        for day_tasks in task_params:
            file_date, day_samples, h5_file = day_tasks
            with h5py.File(h5_file, 'r') as f:
                for _, row in day_samples.iterrows():
                    stock_code = row['stock_code']
                    # 只加载merge后匹配成功的数据
                    features = f[stock_code][:]
                    cache_key = (file_date, stock_code)
                    data_cache[cache_key] = features
                    sample.append({
                        'datetime': file_date,
                        'stock_code': stock_code,
                        'label': row['label']
                    })
        return sample, data_cache
    
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
        date_list = list(merged_by_date.groups.keys())
        chunk_size = len(date_list) // self.num_workers
        samples = []
        all_task_list = [] # 任务列表[]——>[[(file_date, day_samples, h5_file), (...),..],[..]]
        tasks = []
        for i in range(self.num_workers):
            task_list = []
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(date_list))
            each_worker_date = date_list[start_idx:end_idx]
            for file_date in each_worker_date:
                h5_file = h5_file_map[file_date]
                day_samples = merged_by_date.get_group(file_date)
                task_list.append((file_date, day_samples, h5_file))
            all_task_list.append(task_list)

        for one_task in all_task_list:
            tasks.append(delayed(self._eval_one_task)(one_task))

        parall_results = Parallel(n_jobs=self.num_workers)(tqdm(tasks, desc="Processing h5"))
        
        for one_task in parall_results:
            samples_part, data_cache_part = one_task
            samples.extend(samples_part)
            self.data_cache.update(data_cache_part) #这里把不同进程的内存合并到主内存
        
        return samples
        
    def __len__(self):
        return len(self.sample_infos)
    
    def __getitem__(self, idx):
        """从内存中获取特征和标签"""
        sample = self.sample_infos[idx]
        cache_key = (sample['datetime'], sample['stock_code'])
        features = torch.from_numpy(self.data_cache[cache_key]).float()
        
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return {
            'features': features,  # (7, 8, 8)
            'label': label,        # 0, 1, 2
            'stock_code': sample['stock_code'],
            'datetime': sample['datetime']
        }