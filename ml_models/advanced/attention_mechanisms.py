"""
Advanced ML: Attention Mechanisms and Self-Supervised Learning
Enhances model performance through attention and pre-training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class SelfAttention3D(nn.Module):
    """
    3D Self-Attention module for medical image segmentation
    Captures long-range dependencies in 3D volumes
    """
    
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = max(in_channels // reduction, 1)
        
        self.query_conv = nn.Conv3d(in_channels, self.inter_channels, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, self.inter_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, self.inter_channels, kernel_size=1)
        self.out_conv = nn.Conv3d(self.inter_channels, in_channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, D, H, W)
        Returns:
            Attention-enhanced features
        """
        batch_size, C, D, H, W = x.size()
        
        # Query, Key, Value projections
        query = self.query_conv(x).view(batch_size, self.inter_channels, -1)
        key = self.key_conv(x).view(batch_size, self.inter_channels, -1)
        value = self.value_conv(x).view(batch_size, self.inter_channels, -1)
        
        # Attention map
        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, self.inter_channels, D, H, W)
        out = self.out_conv(out)
        
        # Residual connection with learnable weight
        return self.gamma * out + x


class ChannelAttention3D(nn.Module):
    """
    Channel Attention (Squeeze-and-Excitation) for 3D medical images
    Recalibrates channel-wise feature responses
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, C, _, _, _ = x.size()
        
        # Squeeze: Global pooling
        avg_out = self.fc(self.avg_pool(x).view(batch_size, C))
        max_out = self.fc(self.max_pool(x).view(batch_size, C))
        
        # Excitation: Channel-wise attention weights
        attention = self.sigmoid(avg_out + max_out).view(batch_size, C, 1, 1, 1)
        
        return x * attention.expand_as(x)


class SpatialAttention3D(nn.Module):
    """
    Spatial Attention for 3D medical images
    Focuses on informative spatial locations
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and compute attention
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        
        return x * attention


class CBAM3D(nn.Module):
    """
    Convolutional Block Attention Module for 3D
    Combines channel and spatial attention
    """
    
    def __init__(self, in_channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        
        self.channel_attention = ChannelAttention3D(in_channels, reduction)
        self.spatial_attention = SpatialAttention3D(spatial_kernel)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ContrastiveLearning:
    """
    Self-supervised contrastive learning for medical images
    Pre-trains models on unlabeled data
    """
    
    def __init__(self, encoder: nn.Module, temperature: float = 0.07):
        self.encoder = encoder
        self.temperature = temperature
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def nt_xent_loss(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)
        SimCLR loss function
        """
        batch_size = z_i.shape[0]
        
        # Normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Similarity matrix
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), 
                                               representations.unsqueeze(0), 
                                               dim=2)
        
        # Mask diagonal
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
        
        # Positive pairs
        positives = torch.cat([
            torch.diag(similarity_matrix, batch_size),
            torch.diag(similarity_matrix, -batch_size)
        ], dim=0)
        
        # Loss
        nominator = torch.exp(positives / self.temperature)
        denominator = torch.sum(torch.exp(similarity_matrix / self.temperature), dim=1)
        
        loss = -torch.log(nominator / denominator)
        return loss.mean()


class TransformerBlock3D(nn.Module):
    """
    Transformer block for 3D medical image analysis
    Captures global context through self-attention
    """
    
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class AdvancedUNetWithAttention(nn.Module):
    """
    U-Net with attention mechanisms for improved segmentation
    Integrates CBAM and self-attention
    """
    
    def __init__(self, in_channels: int = 4, out_channels: int = 4):
        super().__init__()
        
        # This is a simplified example - would integrate with existing UNet3D
        self.attention_gates = nn.ModuleList([
            CBAM3D(64),
            CBAM3D(128),
            CBAM3D(256)
        ])
        
        self.self_attention = SelfAttention3D(256)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Would integrate with UNet forward pass
        # Apply attention at different scales
        pass


class FeaturePyramidNetwork3D(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature extraction
    Improves detection of tumors at different scales
    """
    
    def __init__(self, in_channels_list: list, out_channels: int = 256):
        super().__init__()
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv3d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])
        
        self.output_convs = nn.ModuleList([
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
    
    def forward(self, features: list) -> list:
        """
        Args:
            features: List of feature maps from different scales
        Returns:
            Multi-scale enhanced features
        """
        # Top-down pathway
        results = []
        last_inner = self.lateral_convs[-1](features[-1])
        results.append(self.output_convs[-1](last_inner))
        
        for idx in range(len(features) - 2, -1, -1):
            inner_lateral = self.lateral_convs[idx](features[idx])
            inner_top_down = F.interpolate(last_inner, size=inner_lateral.shape[-3:], mode='trilinear')
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.output_convs[idx](last_inner))
        
        return results


if __name__ == "__main__":
    print("="*80)
    print("ADVANCED ML: ATTENTION MECHANISMS & SELF-SUPERVISED LEARNING")
    print("="*80)
    
    print("\n✅ Advanced Techniques Implemented:")
    print("   1. Self-Attention for 3D volumes")
    print("   2. Channel Attention (Squeeze-and-Excitation)")
    print("   3. Spatial Attention")
    print("   4. CBAM (Convolutional Block Attention Module)")
    print("   5. Contrastive Self-Supervised Learning")
    print("   6. Transformer blocks for global context")
    print("   7. Feature Pyramid Network for multi-scale features")
    
    print("\n📊 Expected Benefits:")
    print("   • Better focus on tumor regions")
    print("   • Improved boundary detection")
    print("   • Multi-scale feature extraction")
    print("   • Pre-training on unlabeled data")
    print("   • Global context modeling")
    
    print("\n✅ Modules ready for integration!")
