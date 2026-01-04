import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from datetime import datetime

from data.dataset import MockLevel2DatasetViT
from models.vit_regression import ViTRegression
from utils.metrics import calculate_rank_ic, calculate_metrics


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    训练一个 epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前 epoch
    
    Returns:
        avg_loss: 平均损失
        rank_ic: Rank IC
        metrics: 其他指标
    """
    model.train()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (features, labels) in enumerate(pbar):
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(features)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        all_predictions.append(outputs.detach().cpu())
        all_targets.append(labels.detach().cpu())
        
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    avg_loss = total_loss / len(dataloader)
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    rank_ic, _ = calculate_rank_ic(all_predictions, all_targets)
    metrics = calculate_metrics(all_predictions, all_targets)
    
    return avg_loss, rank_ic, metrics


def validate(model, dataloader, criterion, device, epoch):
    """
    验证模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        epoch: 当前 epoch
    
    Returns:
        avg_loss: 平均损失
        rank_ic: Rank IC
        metrics: 其他指标
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Valid]')
        for features, labels in pbar:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            all_predictions.append(outputs.cpu())
            all_targets.append(labels.cpu())
            
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    avg_loss = total_loss / len(dataloader)
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    rank_ic, _ = calculate_rank_ic(all_predictions, all_targets)
    metrics = calculate_metrics(all_predictions, all_targets)
    
    return avg_loss, rank_ic, metrics


def main():
    """
    主训练入口
    """
    print("=" * 80)
    print("ViT/ViViT 股票收益率预测训练流水线")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-4
    weight_decay = 1e-5
    
    print(f"\n配置信息:")
    print(f"  模型类型: ViT")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Weight Decay: {weight_decay}")
    
    print("\n正在加载数据集...")
    train_dataset = MockLevel2DatasetViT(num_samples=2000, seed=42)
    val_dataset = MockLevel2DatasetViT(num_samples=500, seed=123)
    
    model = ViTRegression(
        image_size=8,
        patch_size=2,
        channels=15,
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1,
        pool='cls'
    ).to(device)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"  训练集样本数: {len(train_dataset)}")
    print(f"  验证集样本数: {len(val_dataset)}")
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数量: {num_params:,}")
    
    os.makedirs('checkpoints', exist_ok=True)
    
    best_rank_ic = -float('inf')
    
    print("\n" + "=" * 80)
    print("开始训练")
    print("=" * 80)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*80}")
        
        train_loss, train_rank_ic, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        val_loss, val_rank_ic, val_metrics = validate(
            model, val_loader, criterion, device, epoch
        )
        
        scheduler.step()
        
        print(f"\n训练结果:")
        print(f"  Loss: {train_loss:.6f}")
        print(f"  Rank IC: {train_rank_ic:.4f}")
        print(f"  IC: {train_metrics['ic']:.4f}")
        print(f"  MSE: {train_metrics['mse']:.6f}")
        print(f"  MAE: {train_metrics['mae']:.6f}")
        
        print(f"\n验证结果:")
        print(f"  Loss: {val_loss:.6f}")
        print(f"  Rank IC: {val_rank_ic:.4f}")
        print(f"  IC: {val_metrics['ic']:.4f}")
        print(f"  MSE: {val_metrics['mse']:.6f}")
        print(f"  MAE: {val_metrics['mae']:.6f}")
        
        if val_rank_ic > best_rank_ic:
            best_rank_ic = val_rank_ic
            checkpoint_path = 'checkpoints/best_model_vit.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rank_ic': val_rank_ic,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"\n✓ 保存最佳模型 (Rank IC: {val_rank_ic:.4f}) -> {checkpoint_path}")
        
        if epoch % 5 == 0:
            checkpoint_path = f'checkpoints/model_vit_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rank_ic': val_rank_ic,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  保存检查点 -> {checkpoint_path}")
    
    print("\n" + "=" * 80)
    print("训练完成!")
    print(f"最佳验证 Rank IC: {best_rank_ic:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
