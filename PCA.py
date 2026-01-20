import torch
import torch.nn as nn

class PCA(nn.Module):
    """
    Channel-wise PCA: 將輸入 [B, C, H, W] 降維到 [B, out_dim, H, W]
    空間維度 H, W 保留，只降 channel
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 用線性層做 PCA，權重大小 = [out_dim, in_dim]
        self.proj = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        # x: [B, C, H, W] -> 先 permute 到 [B, H, W, C]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x = self.proj(x)                         # [B, H, W, out_dim]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, out_dim, H, W]
        return x
