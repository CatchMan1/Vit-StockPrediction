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


# ============ 损失函数类 ============

class ICLoss(torch.nn.Module):
    """
    基于 IC (Information Coefficient) 的损失函数
    
    在量化投资中，IC 衡量预测值与真实值之间的线性相关性（Pearson 相关系数）
    损失函数定义为：Loss = -IC(predictions, targets)
    
    优势：
    - 直接优化量化策略关心的指标
    - 对预测值的绝对大小不敏感，只关心相对排序
    - 符合量化选股的实际需求
    """
    
    def __init__(self, eps=1e-8):
        """
        Args:
            eps: 防止除零的小常数
        """
        super(ICLoss, self).__init__()
        self.eps = eps
    
    def forward(self, predictions, targets):
        """
        计算 IC 损失
        
        Args:
            predictions: 模型预测值 (batch_size, 1) 或 (batch_size,)
            targets: 真实标签值 (batch_size, 1) 或 (batch_size,)
        
        Returns:
            loss: -IC（IC 的相反数）
        """
        # 展平为一维张量
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # 计算均值
        pred_mean = torch.mean(predictions)
        target_mean = torch.mean(targets)
        
        # 中心化（减去均值）
        pred_centered = predictions - pred_mean
        target_centered = targets - target_mean
        
        # 计算协方差（分子）
        covariance = torch.mean(pred_centered * target_centered)
        
        # 计算标准差（分母）
        pred_std = torch.sqrt(torch.mean(pred_centered ** 2) + self.eps)
        target_std = torch.sqrt(torch.mean(target_centered ** 2) + self.eps)
        
        # 计算 Pearson 相关系数（IC）
        ic = covariance / (pred_std * target_std + self.eps)
        
        # 返回 IC 的相反数作为损失
        # 最大化 IC <=> 最小化 -IC
        loss = -ic
        
        return loss


class RankICLoss(torch.nn.Module):
    """
    基于 Rank IC (Spearman 秩相关系数) 的损失函数
    
    使用可微分的软排序近似 Spearman 相关系数
    相比 Pearson IC：
    - 对异常值更鲁棒
    - 只关心排序关系，不关心具体数值
    - 更适合股票选股场景
    """
    
    def __init__(self, eps=1e-8, temperature=1.0):
        """
        Args:
            eps: 防止除零的小常数
            temperature: 软排序的温度参数，越小越接近硬排序
        """
        super(RankICLoss, self).__init__()
        self.eps = eps
        self.temperature = temperature
    
    def _soft_rank(self, x):
        """
        可微分的软排序函数
        
        Args:
            x: 输入张量 (batch_size,)
        
        Returns:
            ranks: 软排序结果 (batch_size,)
        """
        n = x.size(0)
        x_expanded = x.unsqueeze(1)  # (n, 1)
        diff = x_expanded - x.unsqueeze(0)  # (n, n)
        
        # 使用 sigmoid 近似阶跃函数
        comparisons = torch.sigmoid(diff / self.temperature)
        
        # 每个元素的排名 = 有多少元素比它小
        ranks = torch.sum(comparisons, dim=1)
        
        return ranks
    
    def forward(self, predictions, targets):
        """
        计算 Rank IC 损失
        
        Args:
            predictions: 模型预测值 (batch_size, 1) 或 (batch_size,)
            targets: 真实标签值 (batch_size, 1) 或 (batch_size,)
        
        Returns:
            loss: -Rank IC
        """
        # 展平为一维张量
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # 计算软排序
        pred_ranks = self._soft_rank(predictions)
        target_ranks = self._soft_rank(targets)
        
        # 计算排序后的 Pearson 相关系数
        pred_mean = torch.mean(pred_ranks)
        target_mean = torch.mean(target_ranks)
        
        pred_centered = pred_ranks - pred_mean
        target_centered = target_ranks - target_mean
        
        covariance = torch.mean(pred_centered * target_centered)
        
        pred_std = torch.sqrt(torch.mean(pred_centered ** 2) + self.eps)
        target_std = torch.sqrt(torch.mean(target_centered ** 2) + self.eps)
        
        rank_ic = covariance / (pred_std * target_std + self.eps)
        
        # 返回 Rank IC 的相反数
        loss = -rank_ic
        
        return loss
