# ViT/ViViT 股票预测训练流水线使用指南

## 项目结构

```
StockPredicton/
├── data/
│   └── dataset.py          # Mock 数据集和真实数据集接口
├── models/
│   ├── __init__.py         # 模型导出
│   ├── vivit.py            # ViViT 基础模型
│   └── vit_regression.py   # ViT/ViViT 回归适配器
├── utils/
│   ├── __init__.py
│   └── metrics.py          # RankIC 和其他评估指标
├── train.py                # 训练脚本
├── evaluate.py             # 评估脚本
├── requirements.txt        # 依赖包
└── README.md               # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 训练模型（使用 Mock 数据）

```bash
python train.py
```

这将使用模拟数据训练 ViViT 模型，默认配置：
- 模型类型: ViViT
- 训练样本: 2000
- 验证样本: 500
- Batch Size: 32
- Epochs: 10
- Learning Rate: 1e-4

训练过程中会自动：
- 计算 Rank IC（核心量化指标）
- 保存最佳模型到 `checkpoints/best_model_vivit.pth`
- 每 5 个 epoch 保存检查点

### 2. 评估模型

```bash
python evaluate.py --checkpoint checkpoints/best_model_vivit.pth --model_type vivit
```

参数说明：
- `--checkpoint`: 模型检查点路径（必需）
- `--model_type`: 模型类型，vivit 或 vit（默认: vivit）
- `--batch_size`: 批次大小（默认: 32）
- `--num_samples`: 测试样本数量（默认: 1000）

## 数据格式

### Mock 数据集

项目提供了两个 Mock 数据集类用于测试：

1. **MockLevel2Dataset** (ViViT 使用)
   - 输入形状: `(20, 15, 8, 8)`
   - 20: 时间序列长度（20个交易日）
   - 15: 特征通道数
   - 8x8: 空间维度

2. **MockLevel2DatasetViT** (ViT 使用)
   - 输入形状: `(15, 8, 8)`
   - 单帧图像特征

### 替换为真实数据

要使用真实的 HDF5 数据，需要在 `data/dataset.py` 中创建新的数据集类：

```python
import h5py

class Level2Dataset(Dataset):
    def __init__(self, h5_path, mode='train'):
        self.h5_path = h5_path
        self.mode = mode
        # 加载 HDF5 数据
        with h5py.File(h5_path, 'r') as f:
            self.features = f[f'{mode}/features'][:]
            self.labels = f[f'{mode}/labels'][:]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        features = torch.from_numpy(self.features[idx]).float()
        label = torch.from_numpy(self.labels[idx]).float()
        return features, label
```

然后在 `train.py` 中替换数据集：

```python
# 替换这部分
train_dataset = Level2Dataset('data/train.h5', mode='train')
val_dataset = Level2Dataset('data/val.h5', mode='val')
```

## 模型配置

### ViViT 模型参数

```python
model = ViViTRegression(
    image_size=8,           # 图像尺寸
    image_patch_size=2,     # 空间 patch 大小
    frames=20,              # 时间序列长度
    frame_patch_size=4,     # 时间 patch 大小
    channels=15,            # 输入通道数
    dim=256,                # Transformer 维度
    spatial_depth=4,        # 空间 Transformer 深度
    temporal_depth=4,       # 时间 Transformer 深度
    heads=8,                # 注意力头数
    mlp_dim=512,            # MLP 隐藏层维度
    dropout=0.1,            # Dropout 率
    emb_dropout=0.1,        # Embedding Dropout 率
    variant='factorized_encoder',  # 变体类型
    pool='cls'              # 池化方式
)
```

### ViT 模型参数

```python
model = ViTRegression(
    image_size=8,           # 图像尺寸
    image_patch_size=2,     # patch 大小
    channels=15,            # 输入通道数
    dim=256,                # Transformer 维度
    depth=6,                # Transformer 深度
    heads=8,                # 注意力头数
    mlp_dim=512,            # MLP 隐藏层维度
    dropout=0.1,            # Dropout 率
    emb_dropout=0.1,        # Embedding Dropout 率
    pool='cls'              # 池化方式
)
```

## 评估指标

### Rank IC (核心指标)
- **定义**: Spearman 秩相关系数，衡量预测排序与真实排序的一致性
- **量化意义**: 在量化策略中，我们更关心预测的相对排序而非绝对值
- **评判标准**:
  - |Rank IC| > 0.05: 模型具有一定预测能力
  - |Rank IC| > 0.10: 模型预测能力较强
  - p-value < 0.05: 统计显著

### 其他指标
- **IC**: Pearson 相关系数
- **MSE**: 均方误差
- **MAE**: 平均绝对误差

## 训练输出示例

```
================================================================================
ViT/ViViT 股票收益率预测训练流水线
================================================================================

使用设备: cuda

配置信息:
  模型类型: VIVIT
  Batch Size: 32
  Epochs: 10
  Learning Rate: 0.0001
  Weight Decay: 1e-05

正在加载数据集...
  训练集样本数: 2000
  验证集样本数: 500

模型参数量: 2,345,678

================================================================================
开始训练
================================================================================

================================================================================
Epoch 1/10
================================================================================
Epoch 1 [Train]: 100%|██████████| 63/63 [00:15<00:00,  4.12it/s, loss=0.002134]
Epoch 1 [Valid]: 100%|██████████| 16/16 [00:02<00:00,  6.45it/s, loss=0.002089]

训练结果:
  Loss: 0.002156
  Rank IC: 0.0234
  IC: 0.0198
  MSE: 0.002156
  MAE: 0.036789

验证结果:
  Loss: 0.002089
  Rank IC: 0.0456
  IC: 0.0423
  MSE: 0.002089
  MAE: 0.035234

✓ 保存最佳模型 (Rank IC: 0.0456) -> checkpoints/best_model_vivit.pth
```

## 注意事项

1. **数据归一化**: Mock 数据已经进行了标准化，真实数据也需要进行类似处理
2. **批次大小**: 根据 GPU 显存调整 batch_size
3. **学习率**: 可以根据训练情况调整学习率和学习率调度器
4. **模型参数**: 可以根据数据规模和计算资源调整模型深度和维度
5. **过拟合**: 注意观察训练集和验证集的 Rank IC 差异，必要时增加 dropout 或正则化

## 下一步工作

1. 准备真实的 Level-2 HDF5 数据
2. 实现真实数据的 Dataset 类
3. 调整模型超参数以适应真实数据
4. 实现回测模块以评估策略收益
5. 添加特征工程和数据增强
