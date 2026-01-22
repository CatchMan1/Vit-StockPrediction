import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import argparse
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from vision_data import H5StockDataset
# from data.dataset_classify import H5StockDataset
# from models.vit_classification import ViTClassification
from model import ViTClassification
import time
from train import train
from infer import infer
params = {
        # 设备参数
        'gpu_id': 5, 
        # 数据参数
        'h5_dir': '/mnt/dataset/songan_space/LV2_vision/vision_data_7channel_4h/',  # H5特征文件目录
        'label_file': "/mnt/dataset/songan_space/input_rq/label_rq_1Dvwap_0p2class_20100101_20251130.csv.gz",  # 标签CSV文件路径
        'train_start_date': '2017-01-01',
        'train_end_date': '2020-11-30',
        'train_val_split': 0.8,
        'infer_start_date': '2021-01-01',
        'infer_end_date': '2024-12-31',
        # 模型参数
        'image_size': 8,
        'patch_size': 2,
        'channels': 21,  
        'dim': 64,  # Transformer嵌入维度
        'depth': 6,  # Transformer层数
        'heads': 8,  # 注意力头数
        'mlp_dim': 256,  # MLP隐藏层维度
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'pool': 'cls',  # cls池化
        'num_classes': 3, 
        # 训练参数
        'batch_size': 256,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'num_workers': 10,
        'random_seed': 42,
        # 保存参数
        # 保存最佳模型
        'checkpoint_dir': 'checkpoints_up',
        #保存推理结果
        'output_dir': 'results_up',
        #保存训练指标
        'metric_dir': 'metrics_up',
        'task_num': 8,
    }
# 保存参数 
checkpoint_dir = params['checkpoint_dir']
output_dir = params['output_dir']
metric_dir = params['metric_dir']
p_cls = f"_dim{params['dim']}_mlp{params['mlp_dim']}_depth{params['depth']}_heads{params['heads']}"
p_rate = f"_lr{params['learning_rate']}_dr{params['dropout']}_emb{params['emb_dropout']}_dc{params['weight_decay']}"
params['parameters_classification'] = p_cls
params['parameters_rate'] = p_rate
# 保存最佳模型
params['checkpoint_classification_dir'] = f'{checkpoint_dir}/Vit_classification{p_cls}'
params['checkpoint_path'] = f'{checkpoint_dir}/Vit_classification{p_cls}/best_model_{p_rate}.pth'
# 保存推理结果
params['output_classification_dir'] = f'{output_dir}/Vit_classification{p_cls}'
params['infer_path'] = f'{output_dir}/Vit_classification{p_cls}/{p_rate}.csv'
params['infer_path_gz'] = f'{output_dir}/Vit_classification{p_cls}/{p_rate}.csv.gz'
params['output_task'] = f'{output_dir}/Vit_classification{p_cls}/task{params["task_num"]}.csv'
# 保存训练指标
params['metric_classification_dir'] = f'{metric_dir}/Vit_classification{p_cls}'
params['curves_path'] = f'{metric_dir}/Vit_classification{p_cls}/training_curves_{p_rate}.png'
params['train_csv_path'] = f'{metric_dir}/Vit_classification{p_cls}/training_metrics_{p_rate}.csv'
params['train_json_path'] = f'{metric_dir}/Vit_classification{p_cls}/training_metrics_{p_rate}.json'

def main():
    train(params)
    infer(params)

if __name__ == '__main__':
    main()