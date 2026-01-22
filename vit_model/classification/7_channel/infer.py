from calendar import day_abbr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import re
import pandas as pd
import h5py
import os
import argparse
from datetime import datetime
from pathlib import Path
# from data.dataset_test import H5StockDataset
from vision_data import H5StockDataset
# from models.vit_classification import ViTClassification
from model import ViTClassification

def infer(params):
    
    device = torch.device(f"cuda:{params['gpu_id']}")
    torch.cuda.set_device(params['gpu_id'])
    ## 加载模型
    checkpoint = torch.load(params['checkpoint_path'], map_location=device)
    
    saved_params = checkpoint['params']
    # 只更新模型相关参数，保留当前的数据和推理参数
    for key in ['image_size', 'patch_size', 'channels', 'dim', 'depth', 'heads', 'mlp_dim', 'dropout', 'emb_dropout', 'pool', 'num_classes']:
        if key in saved_params:
            params[key] = saved_params[key]
    print("  使用checkpoint中保存的模型参数")
    
    model = ViTClassification(
        image_size=params['image_size'],
        patch_size=params['patch_size'],
        channels=params['channels'],
        dim=params['dim'],
        depth=params['depth'],
        heads=params['heads'],
        mlp_dim=params['mlp_dim'],
        dropout=params['dropout'],
        emb_dropout=params['emb_dropout'],
        pool=params['pool'],
        num_classes=params['num_classes']
    ).to(device)
    factor_list = []
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    h5_file_path = Path(params['h5_dir'])
    start_date = params['infer_start_date']
    end_date = params['infer_end_date']
    all_h5 = sorted(h5_file_path.glob('*.h5'))
    day_list = []
    for h5_file in all_h5:
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', h5_file.name)
        # 检查文件名是否包含日期
        file_date = date_match.group(1)
        if start_date <= file_date <= end_date:
            day_info = {}  # 每个日期创建独立的字典
            with h5py.File(h5_file, 'r') as f:
                for stock_code in f.keys():
                    # 读取并归一化
                    features = f[stock_code][:].astype(np.float32)
                    for c in range(features.shape[0]):
                        channel_max = np.max(np.abs(features[c]))
                        if channel_max > 0:
                            features[c] /= channel_max
                    day_info[stock_code] = features
            day_list.append((file_date, day_info))

    for file_date, day_info in tqdm(day_list, desc='VIT-Inference'):
        outputs_list = []
        infer_data_tensor = torch.stack([
            torch.from_numpy(day_info[key]).float() for key in day_info.keys()
        ])
        infer_data_tensor = infer_data_tensor.to(device)
        with torch.no_grad():
            num_batches = (len(infer_data_tensor) + params['batch_size'] - 1) // params['batch_size']
            # 在一个日期内对batch_size进行划分,对一批一批股票进行模型推理
            for i in range(num_batches):
                batch_size = params['batch_size']
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(infer_data_tensor))
                batch_tensor = infer_data_tensor[start_idx:end_idx].to(device)
                outputs_batch = model(batch_tensor).float()
                prob = F.softmax(outputs_batch, dim=1)
                outputs_list.append(prob.cpu())
            all_prob = torch.cat(outputs_list, dim=0)
            factor = all_prob[:, 2] - all_prob[:, 0]
            data_list = factor.tolist()
            day_df = pd.DataFrame([[file_date] + data_list], columns=['date'] + list(day_info.keys()))
            factor_list.append(day_df)
    factor_df = pd.concat(factor_list)
    os.makedirs(params['output_dir'], exist_ok=True)
    os.makedirs(params['output_classification_dir'], exist_ok=True)
    output_file = params['infer_path']
    output_gz = params['infer_path_gz']
    output_task = params['output_task']
    factor_df.to_csv(output_file, index = False)
    factor_df.to_csv(output_gz, index = False)
    factor_df.to_csv(output_task, index = False)