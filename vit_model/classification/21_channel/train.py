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
import matplotlib.pyplot as plt
import json
import pandas as pd

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]', mininterval=50)
    for batch_idx, batch in enumerate(pbar):
        features = batch['features'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(features)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(outputs, dim=1)
        all_predictions.append(preds.detach().cpu())
        all_targets.append(labels.detach().cpu())

        if batch_idx % 50 == 0 or batch_idx == len(dataloader) - 1:
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    avg_loss = total_loss / len(dataloader)
    
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='macro', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return avg_loss, accuracy, metrics


def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Valid]', mininterval=30)
        for batch_idx,batch in enumerate(pbar):
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(features)
            
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_predictions.append(preds.cpu())
            all_targets.append(labels.cpu())
            
            if batch_idx % 30 == 0 or batch_idx == len(dataloader) - 1:
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    avg_loss = total_loss / len(dataloader)
    
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='macro', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    
    return avg_loss, accuracy, metrics

def save_training_metrics(metrics_history, params):
        # 训练指标
        os.makedirs(params['metric_dir'], exist_ok=True)
        os.makedirs(params['metric_classification_dir'], exist_ok=True)
        # 保存为JSON文件
        json_path = params['train_json_path']
        with open(json_path, 'w') as f:
            json.dump(metrics_history, f, indent=2)
        print(f"指标数据已保存到: {json_path}")
        
        # 保存为CSV文件
        df = pd.DataFrame(metrics_history)
        csv_path = params['train_csv_path']
        df.to_csv(csv_path, index=False)
        print(f"指标数据已保存为CSV: {csv_path}")


def plot_training_curves(metrics_history, params):
    # 绘制训练和验证的损失与准确率曲线
    os.makedirs(params['metric_dir'], exist_ok=True)
    os.makedirs(params['metric_classification_dir'], exist_ok=True)
    epochs = metrics_history['epoch']
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制损失曲线
    ax1.plot(epochs, metrics_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, metrics_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制准确率曲线
    ax2.plot(epochs, metrics_history['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, metrics_history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = params['curves_path']
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线图已保存到: {plot_path}")

def train(params):
    
    device = torch.device(f"cuda:{params['gpu_id']}")
    torch.cuda.set_device(params['gpu_id'])
    print(f"\n配置信息:")
    print(f"  模型类型: ViT Classification")
    print(f"  Batch Size: {params['batch_size']}")
    print(f"  Epochs: {params['num_epochs']}")
    print(f"  Learning Rate: {params['learning_rate']}")
    print(f"  Weight Decay: {params['weight_decay']}")
    print(f"  训练数据: {params['train_start_date']} ~ {params['train_end_date']}")
    print("\n正在加载数据集...")
    start_time = time.time()
    train_dataset = H5StockDataset(
        h5_dir=params['h5_dir'],
        label_dir=params['label_file'],
        start_date=params['train_start_date'],
        end_date=params['train_end_date'],
        num_workers=params['num_workers']
    )
    
    train_size = int(params['train_val_split'] * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(params['random_seed'])
    )
    
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=params['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=params['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    end_time = time.time()
    print(f"数据集加载完成，耗时: {end_time - start_time:.2f}秒")
    print(f"  训练集样本数: {len(train_dataset)}")
    print(f"  验证集样本数: {len(val_dataset)}")
    criterion = nn.CrossEntropyLoss()
    print(f"\n损失函数: Cross Entropy Loss")
    
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['num_epochs'])
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量: {num_params:,}")
    
    os.makedirs(params['checkpoint_dir'], exist_ok=True)
    os.makedirs(params['checkpoint_classification_dir'], exist_ok=True)
    # 初始化指标记录 

    metrics_history = {
        'epoch': [],
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    best_accuracy = 0.0
    patience = 10
    no_improve_count = 0
    
    print("开始训练")
    print(f"早停设置: 验证准确率连续{patience}次无提升将停止训练")
    for epoch in range(1, params['num_epochs'] + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{params['num_epochs']}")
        print(f"{'='*80}")
        
        train_loss, train_acc, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc, _ = validate(
            model, val_loader, criterion, device, epoch
        )
        scheduler.step()
        print(f"\n训练结果:")
        print(f"  Loss: {train_loss:.6f}")
        print(f"  Accuracy: {train_acc:.4f}")
        
        print(f"\n验证结果:")
        print(f"  Loss: {val_loss:.6f}")
        print(f"  Accuracy: {val_acc:.4f}")
        metrics_history['epoch'].append(epoch)
        metrics_history['train_loss'].append(float(train_loss))
        metrics_history['train_accuracy'].append(float(train_acc))
        metrics_history['val_loss'].append(float(val_loss))
        metrics_history['val_accuracy'].append(float(val_acc))
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            no_improve_count = 0
            checkpoint_path = params['checkpoint_path']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'val_loss': val_loss,
                'params': params,
            }, checkpoint_path)
            print(f"\n 保存最佳模型 (Accuracy: {val_acc:.4f}) -> {checkpoint_path}")
        else:
            no_improve_count += 1
            print(f"\n 验证准确率未提升，当前无改善次数: {no_improve_count}/{patience}")
            
            if no_improve_count >= patience:
                print(f"\n早停触发: 验证准确率连续{patience}次未提升，停止训练")
                break
        
    print("训练结束")
    print(f"最佳验证准确率: {best_accuracy:.4f}")
    print(f"实际训练轮数: {epoch}/{params['num_epochs']}")
    save_training_metrics(metrics_history, params)
    plot_training_curves(metrics_history, params)

