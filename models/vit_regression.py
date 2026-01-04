import torch
import torch.nn as nn
from .vit import ViT


class ViTRegression(nn.Module):
    """
    基于 ViT 的回归模型，用于预测股票收益率
    将原始的分类头替换为回归头
    """
    def __init__(
        self,
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
    ):
        """
        Args:
            image_size: 图像尺寸 (8x8)
            patch_size: patch 大小
            channels: 输入通道数 (15)
            dim: Transformer 维度
            depth: Transformer 深度
            heads: 注意力头数
            mlp_dim: MLP 隐藏层维度
            dropout: Dropout 率
            emb_dropout: Embedding Dropout 率
            pool: 'cls' 或 'mean'
        """
        super().__init__()
        
        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=1,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool=pool,
            channels=channels,
            dim_head=64,
            dropout=dropout,
            emb_dropout=emb_dropout
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width) = (B, 15, 8, 8)
        Returns:
            predictions: (batch, 1) 预测的收益率
        """
        output = self.vit(x)
        return output
