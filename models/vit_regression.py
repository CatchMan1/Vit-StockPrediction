# Vision Transformer 回归模型
# 用于股票收益率预测的 ViT 改进版本
# 将原始 ViT 的分类任务改为回归任务

import torch
import torch.nn as nn
from .vit import ViT


class ViTRegression(nn.Module):
    """
    基于 ViT 的回归模型，用于预测股票收益率
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
        pool='cls'# 池化采用平均池化还是CLS token
    ):
        super().__init__()
        
        # 创建 ViT 主干网络
        # num_classes=1，将分类问题转换为回归问题
        self.vit = ViT(
            image_size=image_size,      # 特征图尺寸
            patch_size=patch_size,      # Patch 大小
            num_classes=1,              # 输出维度=1（回归任务）
            dim=dim,                    # Transformer 嵌入维度
            depth=depth,                # Transformer 层数
            heads=heads,                # 注意力头数
            mlp_dim=mlp_dim,           # MLP 隐藏层维度
            pool=pool,                  # 池化方式
            channels=channels,          # 输入通道数（15个股票特征）
            dim_head=64,               # 每个注意力头的维度（固定为64）
            dropout=dropout,            # Dropout 概率
            emb_dropout=emb_dropout    # Embedding Dropout 概率
        )
        
        # 注意：这里直接使用 ViT 的 mlp_head 作为回归头
        # mlp_head 是一个简单的线性层：nn.Linear(dim, 1)
        # 输出是未经激活的原始值，可以是任意实数（适合回归任务）
        
    def forward(self, x):
        # 直接调用 ViT 的前向传播
        # ViT 内部会处理所有的 patch embedding、位置编码、Transformer 编码和池化
        output = self.vit(x)  # (batch, 1)
        return output