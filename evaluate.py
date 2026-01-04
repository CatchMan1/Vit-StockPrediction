import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import os

from data.dataset import MockLevel2DatasetViT
from models.vit_regression import ViTRegression
from utils.metrics import calculate_rank_ic, calculate_metrics


def evaluate_model(model, dataloader, device):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        dataloader: 测试数据加载器
        device: 设备
    
    Returns:
        metrics: 评估指标字典
        predictions: 所有预测值
        targets: 所有真实值
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    print("\n正在评估模型...")
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc='评估进度'):
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            
            all_predictions.append(outputs.cpu())
            all_targets.append(labels.cpu())
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    metrics = calculate_metrics(all_predictions, all_targets)
    
    return metrics, all_predictions, all_targets


def print_metrics(metrics):
    """
    打印评估指标
    
    Args:
        metrics: 指标字典
    """
    print("\n" + "=" * 80)
    print("评估结果")
    print("=" * 80)
    print(f"\n核心量化指标:")
    print(f"  Rank IC (Spearman):  {metrics['rank_ic']:.4f}  (p-value: {metrics['rank_ic_pvalue']:.4e})")
    print(f"  IC (Pearson):        {metrics['ic']:.4f}  (p-value: {metrics['ic_pvalue']:.4e})")
    
    print(f"\n回归指标:")
    print(f"  MSE (均方误差):      {metrics['mse']:.6f}")
    print(f"  MAE (平均绝对误差):  {metrics['mae']:.6f}")
    
    print("\n" + "=" * 80)
    
    if abs(metrics['rank_ic']) > 0.05:
        print("✓ Rank IC 绝对值 > 0.05，模型具有一定的预测能力")
    else:
        print("✗ Rank IC 绝对值 < 0.05，模型预测能力较弱")
    
    if metrics['rank_ic_pvalue'] < 0.05:
        print("✓ Rank IC 显著性检验通过 (p < 0.05)")
    else:
        print("✗ Rank IC 显著性检验未通过 (p >= 0.05)")
    
    print("=" * 80)


def main():
    """
    主评估入口
    """
    parser = argparse.ArgumentParser(description='评估 ViT 股票预测模型')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_samples', type=int, default=1000, help='测试样本数量')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ViT 股票收益率预测模型评估")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    print(f"模型类型: ViT")
    print(f"检查点路径: {args.checkpoint}")
    
    if not os.path.exists(args.checkpoint):
        print(f"\n错误: 检查点文件不存在: {args.checkpoint}")
        return
    
    print("\n正在加载模型...")
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
    
    test_dataset = MockLevel2DatasetViT(num_samples=args.num_samples, seed=999)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ 成功加载检查点 (Epoch: {checkpoint.get('epoch', 'N/A')})")
    if 'val_rank_ic' in checkpoint:
        print(f"  验证集 Rank IC: {checkpoint['val_rank_ic']:.4f}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\n测试集样本数: {len(test_dataset)}")
    
    metrics, predictions, targets = evaluate_model(model, test_loader, device)
    
    print_metrics(metrics)
    
    print("\n预测值统计:")
    pred_np = predictions.numpy().flatten()
    print(f"  均值: {pred_np.mean():.6f}")
    print(f"  标准差: {pred_np.std():.6f}")
    print(f"  最小值: {pred_np.min():.6f}")
    print(f"  最大值: {pred_np.max():.6f}")
    
    print("\n真实值统计:")
    target_np = targets.numpy().flatten()
    print(f"  均值: {target_np.mean():.6f}")
    print(f"  标准差: {target_np.std():.6f}")
    print(f"  最小值: {target_np.min():.6f}")
    print(f"  最大值: {target_np.max():.6f}")


if __name__ == '__main__':
    main()
