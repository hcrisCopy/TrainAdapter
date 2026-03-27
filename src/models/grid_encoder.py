"""
Grid Encoder: 轻量级CNN，从网格图像提取网格特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """残差块，用于构建更深的网络"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class GridEncoder(nn.Module):
    """
    轻量级CNN网格编码器
    输入: [B, 3, H, W] 网格图像
    输出: [B, D, h, w] 网格特征图
    """
    def __init__(self, input_channels=3, feature_dim=512):
        super(GridEncoder, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(input_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # 残差块组
        self.layer1 = self._make_layer(64, 64, 2, stride=1)   # [B, 64, H/4, W/4]
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # [B, 128, H/8, W/8]
        self.layer3 = self._make_layer(128, 256, 2, stride=2) # [B, 256, H/16, W/16]
        self.layer4 = self._make_layer(256, feature_dim, 2, stride=2) # [B, D, H/32, W/32]
        
        # 特征维度
        self.feature_dim = feature_dim
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 网格图像 [B, 3, H, W]
        Returns:
            网格特征 [B, D, h, w]
        """
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, H/2, W/2]
        x = self.maxpool(x)                  # [B, 64, H/4, W/4]
        
        x = self.layer1(x)                   # [B, 64, H/4, W/4]
        x = self.layer2(x)                   # [B, 128, H/8, W/8]
        x = self.layer3(x)                   # [B, 256, H/16, W/16]
        x = self.layer4(x)                   # [B, D, H/32, W/32]
        
        return x


class FeatureProjector(nn.Module):
    """
    特征投影器：将网格特征图转换为与视觉特征匹配的token序列
    输入: [B, D, h, w] 网格特征图
    输出: [B, M, D] 网格token序列
    """
    def __init__(self, input_dim, output_dim, num_tokens=64):
        super(FeatureProjector, self).__init__()
        
        self.num_tokens = num_tokens
        self.output_dim = output_dim
        
        # Adaptive pooling到固定空间尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((int(num_tokens**0.5), int(num_tokens**0.5)))
        
        # 投影到输出维度
        self.projection = nn.Linear(input_dim, output_dim)
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, output_dim))
        
        # LayerNorm
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: 网格特征图 [B, D, h, w]
        Returns:
            网格token序列 [B, M, D]
        """
        B, D, h, w = x.shape
        
        # Adaptive pooling
        x = self.adaptive_pool(x)  # [B, D, sqrt(M), sqrt(M)]
        
        # 展平
        x = x.view(B, D, -1).transpose(1, 2)  # [B, M, D]
        
        # 投影
        x = self.projection(x)  # [B, M, output_dim]
        
        # 添加位置编码
        x = x + self.pos_embedding
        
        # LayerNorm
        x = self.norm(x)
        
        return x


if __name__ == "__main__":
    # 测试代码
    grid_encoder = GridEncoder(input_channels=3, feature_dim=512)
    projector = FeatureProjector(input_dim=512, output_dim=768, num_tokens=64)
    
    # 模拟输入
    grid_image = torch.randn(2, 3, 224, 224)
    
    # 前向传播
    grid_features = grid_encoder(grid_image)  # [2, 512, 7, 7]
    grid_tokens = projector(grid_features)     # [2, 64, 768]
    
    print(f"Grid features shape: {grid_features.shape}")
    print(f"Grid tokens shape: {grid_tokens.shape}")
