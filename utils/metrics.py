import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr


def calculate_rank_ic(predictions, targets):
    """
    计算 Rank IC (Spearman 秩相关系数)
    这是量化策略中的核心评估指标
    
    Args:
        predictions: 预测值，shape (N,) 或 (N, 1)
        targets: 真实值，shape (N,) 或 (N, 1)
    
    Returns:
        rank_ic: Spearman 秩相关系数
        p_value: 显著性 p 值
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    if len(predictions) < 2:
        return 0.0, 1.0
    
    rank_ic, p_value = spearmanr(predictions, targets)
    
    if np.isnan(rank_ic):
        rank_ic = 0.0
    
    return rank_ic, p_value


def calculate_ic(predictions, targets):
    """
    计算 IC (Pearson 相关系数)
    
    Args:
        predictions: 预测值，shape (N,) 或 (N, 1)
        targets: 真实值，shape (N,) 或 (N, 1)
    
    Returns:
        ic: Pearson 相关系数
        p_value: 显著性 p 值
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    if len(predictions) < 2:
        return 0.0, 1.0
    
    ic, p_value = pearsonr(predictions, targets)
    
    if np.isnan(ic):
        ic = 0.0
    
    return ic, p_value


def calculate_metrics(predictions, targets):
    """
    计算多个评估指标
    
    Args:
        predictions: 预测值
        targets: 真实值
    
    Returns:
        metrics: 包含各种指标的字典
    """
    rank_ic, rank_p = calculate_rank_ic(predictions, targets)
    ic, ic_p = calculate_ic(predictions, targets)
    
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    metrics = {
        'rank_ic': rank_ic,
        'rank_ic_pvalue': rank_p,
        'ic': ic,
        'ic_pvalue': ic_p,
        'mse': mse,
        'mae': mae
    }
    
    return metrics
