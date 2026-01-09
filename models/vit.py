# Vision Transformer (ViT) 实现
# 用于图像分类任务的 Transformer 架构

import torch
from torch import nn
from torch.nn import Module, ModuleList

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# ============ 辅助函数 ============

def pair(t):
    """
    将输入转换为元组对
    如果输入已经是元组则直接返回，否则将单个值转换为 (t, t)
    用于处理图像尺寸和patch尺寸参数
    """
    return t if isinstance(t, tuple) else (t, t)

# ============ 核心模块类 ============

class FeedForward(Module):
    """
    前馈神经网络 (Feed-Forward Network)
    Transformer 中的 MLP 部分，用于特征变换
    结构: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout
    """
    def __init__(self, dim, hidden_dim, dropout = 0.):
        """
        Args:
            dim: 输入和输出的特征维度
            hidden_dim: 隐藏层维度（通常是 dim 的 4 倍）
            dropout: Dropout 概率
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),              # 层归一化
            nn.Linear(dim, hidden_dim),      # 升维：dim -> hidden_dim
            nn.GELU(),                       # GELU 激活函数（比 ReLU 更平滑）
            nn.Dropout(dropout),             # Dropout 正则化
            nn.Linear(hidden_dim, dim),      # 降维：hidden_dim -> dim
            nn.Dropout(dropout)              # Dropout 正则化
        )

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量 (batch, seq_len, dim)
        Returns:
            输出张量 (batch, seq_len, dim)
        """
        return self.net(x)

class Attention(Module):
    """
    多头自注意力机制 (Multi-Head Self-Attention)
    Transformer 的核心组件，用于捕捉序列中不同位置之间的关系
    """
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        """
        Args:
            dim: 输入特征维度
            heads: 注意力头的数量
            dim_head: 每个注意力头的维度
            dropout: Dropout 概率
        """
        super().__init__()
        inner_dim = dim_head *  heads  # 所有头的总维度
        project_out = not (heads == 1 and dim_head == dim)  # 判断是否需要输出投影

        self.heads = heads
        self.scale = dim_head ** -0.5  # 缩放因子，用于缩放点积注意力（1/sqrt(d_k)）

        self.norm = nn.LayerNorm(dim)  # 层归一化

        self.attend = nn.Softmax(dim = -1)  # Softmax 用于计算注意力权重
        self.dropout = nn.Dropout(dropout)

        # 一次性生成 Q、K、V 三个矩阵（效率更高）
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)# dim=256时，inner_dim*3的维度为192时，拆除Q,K,V权重矩阵分别为(64, 256),dim->x, inner_dim*3->y

        # 输出投影层：将多头的输出合并回原始维度，便于统一维度后续计算
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),# 默认对最后一个维度进行处理
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """
        前向传播：计算多头自注意力
        Args:
            x: 输入张量 (batch, seq_len, dim)
        Returns:
            输出张量 (batch, seq_len, dim)
        """
        x = self.norm(x)  # 先进行层归一化

        # 生成 Q、K、V 并分割成三个张量
        qkv = self.to_qkv(x).chunk(3, dim = -1)# 拆分数据张量
        # 重排维度：(batch, seq_len, heads*dim_head) -> (batch, heads, seq_len, dim_head)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)# 多头自注意力共享权重更新，只是通过视图变换得到单注意力

        # 计算每个注意力分数：Q @ K^T / sqrt(d_k),得到结果矩阵(b, h, n, n)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale# 多头自注意力这块除以的是dim_head**1/2, 如果是单头的话则是inner_dim**1/2

        # 应用 Softmax 得到注意力权重
        attn = self.attend(dots)# 对最后一行的注意力分数转换成（0~1)的概率分布
        attn = self.dropout(attn)  # Dropout 正则化

        # 使用注意力权重对 V 进行加权求和
        out = torch.matmul(attn, v)
        # 重排维度：(batch, heads, seq_len, dim_head) -> (batch, seq_len, heads*dim_head)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)  # 通过输出投影层

class Transformer(Module):
    """
    Transformer 编码器
    由多层 Attention + FeedForward 组成，使用残差连接
    """
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        """
        Args:
            dim: 特征维度
            depth: Transformer 层数
            heads: 注意力头数
            dim_head: 每个注意力头的维度
            mlp_dim: MLP 隐藏层维度
            dropout: Dropout 概率
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 最终的层归一化
        self.layers = ModuleList([])   # 存储所有 Transformer 层

        # 构建 depth 层 Transformer Block
        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),  # 注意力层
                FeedForward(dim, mlp_dim, dropout = dropout)  # 前馈网络层
            ]))

    def forward(self, x):
        """
        前向传播：依次通过所有 Transformer 层
        Args:
            x: 输入张量 (batch, seq_len, dim)
        Returns:
            输出张量 (batch, seq_len, dim)
        """
        # 遍历每一层 Transformer Block
        for attn, ff in self.layers:
            x = attn(x) + x  # 注意力层 + 残差连接
            x = ff(x) + x    # 前馈网络层 + 残差连接

        return self.norm(x)  # 最终的层归一化

class ViT(Module):
    """
    Vision Transformer (ViT) 主模型
    将图像分割成 patches，通过 Transformer 编码器处理，最后进行分类
    
    核心思想：
    1. 将图像分割成固定大小的 patches
    2. 将每个 patch 线性投影到嵌入空间
    3. 添加位置编码
    4. 通过 Transformer 编码器处理
    5. 使用 [CLS] token 或平均池化进行分类
    """
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        """
        Args:
            image_size: 图像尺寸（高度, 宽度）或单个整数
            patch_size: patch 尺寸（高度, 宽度）或单个整数
            num_classes: 分类类别数
            dim: Transformer 的特征维度
            depth: Transformer 层数
            heads: 注意力头数
            mlp_dim: MLP 隐藏层维度
            pool: 池化方式 ('cls' 使用 CLS token, 'mean' 使用平均池化)
            channels: 输入图像通道数（RGB=3）
            dim_head: 每个注意力头的维度
            dropout: Transformer 中的 Dropout 概率
            emb_dropout: Embedding 层的 Dropout 概率
        """
        super().__init__()
        # 将图像尺寸和 patch 尺寸转换为 (height, width) 元组
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # 确保图像尺寸可以被 patch 尺寸整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # 计算 patch 的数量和每个 patch 的维度
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width  # 展平后的 patch 维度

        # 验证池化方式
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_cls_tokens = 1 if pool == 'cls' else 0  # 是否使用 CLS token
        """
        # Patch Embedding 层：将图像分割成 patches 并投影到嵌入空间
        # 重排维度：(batch, channels, height, width) -> (batch, num_patches, patch_dim)
        #   b: batch size（批大小）
        #   c: channels（通道数，RGB=3）
        #   h: 垂直方向的 patch 数量
        #   w: 水平方向的 patch 数量
        #   p1: 每个 patch 的高度（patch_height）
        #   p2: 每个 patch 的宽度（patch_width）
        """
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),      # 归一化（在15个通道层面进行patch归一化）
            nn.Linear(patch_dim, dim),     # 线性投影到 Transformer 维度
            nn.LayerNorm(dim),             # 再次归一化
        )

        # CLS token：用于分类的特殊 token（如果使用 cls 池化）
        self.cls_token = nn.Parameter(torch.randn(num_cls_tokens, dim))
        # 位置编码：为每个 patch 添加位置信息（可学习参数）
        self.pos_embedding = nn.Parameter(torch.randn(num_patches + num_cls_tokens, dim))

        self.dropout = nn.Dropout(emb_dropout)  # Embedding 层的 Dropout

        # Transformer 编码器
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool  # 池化方式
        self.to_latent = nn.Identity()  # 潜在空间映射（这里是恒等映射）

        # 分类头：将 Transformer 输出映射到类别数
        self.mlp_head = nn.Linear(dim, num_classes)

    # 率先执行
    def forward(self, img):
        """
        前向传播：完整的 ViT 推理流程
        Args:
            img: 输入图像 (batch, channels, height, width)
        Returns:
            分类 logits (batch, num_classes)
        """
        batch = img.shape[0] # 截面预测，batch=5000，即5000个股票
        # 步骤1: 将图像转换为 patch embeddings
        x = self.to_patch_embedding(img)  # 将图像从(batch, channel, H, W)转换为(batch, num_patches, dim)

        # 步骤2: 添加 CLS token（如果使用 cls 池化）
        cls_tokens = repeat(self.cls_token, '... d -> b ... d', b = batch)  # 复制到 batch 维度.(1,dim)-->(batch,1,dim)
        x = torch.cat((cls_tokens, x), dim = 1)  # 在序列开头拼接 CLS token, (batch,1,dim)+(batch,num_patches,dim)-->(batch,num_patches+1,dim)

        seq = x.shape[1]  # 序列长度（num_patches + num_cls_tokens）

        # 步骤3: 添加位置编码
        x = x + self.pos_embedding[:seq]  # 广播加法
        x = self.dropout(x)  # Dropout 正则化

        # 步骤4: 通过 Transformer 编码器
        x = self.transformer(x)  # (batch, seq_len, dim)

        # 步骤5: 池化操作(两种)
        # 'mean': 对所有 tokens 取平均
        # 'cls': 只使用 CLS token（第一个 token）
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # 步骤6: 通过分类头得到最终输出
        x = self.to_latent(x)  # 潜在空间映射（这里是恒等映射）
        return self.mlp_head(x)  # (batch, num_classes)
