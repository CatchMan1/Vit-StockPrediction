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

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
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
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Valid]')
        for batch in pbar:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(features)
            
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_predictions.append(preds.cpu())
            all_targets.append(labels.cpu())
            
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
    
    best_accuracy = 0.0
    
    print("开始训练")
    for epoch in range(1, params['num_epochs'] + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{params['num_epochs']}")
        print(f"{'='*80}")
        
        train_loss, train_acc, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc, val_metrics = validate(
            model, val_loader, criterion, device, epoch
        )
        scheduler.step()
        print(f"\n训练结果:")
        print(f"  Loss: {train_loss:.6f}")
        print(f"  Accuracy: {train_acc:.4f}")
        print(f"  Precision: {train_metrics['precision']:.4f}")
        print(f"  Recall: {train_metrics['recall']:.4f}")
        print(f"  F1 Score: {train_metrics['f1']:.4f}")
        
        print(f"\n验证结果:")
        print(f"  Loss: {val_loss:.6f}")
        print(f"  Accuracy: {val_acc:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        print(f"  F1 Score: {val_metrics['f1']:.4f}")
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            checkpoint_path = os.path.join(params['checkpoint_dir'], 'best_model_vit_classification.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'val_loss': val_loss,
                'params': params,
            }, checkpoint_path)
            print(f"\n 保存最佳模型 (Accuracy: {val_acc:.4f}) -> {checkpoint_path}")
        
        if epoch % params['save_interval'] == 0:
            checkpoint_path = os.path.join(params['checkpoint_dir'], f'model_vit_classification_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'val_loss': val_loss,
                'params': params,
            }, checkpoint_path)
            print(f"  保存检查点 -> {checkpoint_path}")
    print("训练结束")
    print(f"最佳验证准确率: {best_accuracy:.4f}")
