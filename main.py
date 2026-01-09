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
        'train_end_date': '2017-12-31',
        'train_val_split': 0.8,
        'infer_start_date': '2021-01-01',
        'infer_end_date': '2021-12-31',
        # 模型参数
        'image_size': 8,
        'patch_size': 2,
        'channels': 7,  
        'dim': 256,  # Transformer嵌入维度
        'depth': 6,  # Transformer层数
        'heads': 8,  # 注意力头数
        'mlp_dim': 512,  # MLP隐藏层维度
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'pool': 'cls',  # cls池化
        'num_classes': 3, 
        # 训练参数
        'batch_size': 64,
        'num_epochs': 3,
        'learning_rate': 0.0001,
        'weight_decay': 1e-5,
        'num_workers': 10,
        'random_seed': 42,
        # 保存参数
        'checkpoint_dir': 'checkpoints',
        'checkpoint_path': 'checkpoints/best_model_vit_classification.pth',
        'output_dir': 'results',
    }

def main():
    infer(params)

if __name__ == '__main__':
    main()