import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEncoderCNN(nn.Module):
    """
    改良版 PatchEncoderCNN
    功能:
        - 下採樣壓縮空間
        - 1x1 Conv 在 pooling 前做 channel 互動
        - 輸出: [B, 1, out_dim]
    """

    def __init__(self, in_channels=180, num_downsample=4):
        """
        in_channels: 輸入 feature channel
        out_dim: latent vector 維度
        hidden_channels: CNN 中間 channel
        num_downsample: 下採樣次數
        """
        super().__init__()
        layers = []

        for i in range(num_downsample):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)


    def forward(self, x):
        """
        x: [B, in_channels, H, W]
        out: [B, 1, out_dim]
        """
        x = self.encoder(x)           # 下採樣 + ReLU
        x = x.view(x.size(0), 1, -1)
        #print("x.shape", x.shape)
        return x
