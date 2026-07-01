import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import os
from torchvision import transforms
import time
from tqdm import tqdm
from torchvision.utils import save_image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 訓練參數
num_epochs = 1000
learning_rate = 1e-3
image_size = 256  # 圖片大小 (H*W)
batch_size = 1  # 一次訓練一張圖片
num_gaussians = 1000  # 高斯總數

# 2DGS 參數數量
# [x, y, scale, rotation, r, g, b, opacity]
gs_param_dim = 8


class GaussianParameterModel(nn.Module):
    """2DGS 參數模型 - num_gaussians 個高斯點"""
    def __init__(self, num_gaussians, gs_param_dim):
        super(GaussianParameterModel, self).__init__()
        self.num_gaussians = num_gaussians
        self.gs_param_dim = gs_param_dim
        
        # 初始化 2DGS 參數
        # Shape: [num_gaussians, gs_param_dim]
        # gs_param_dim: [x, y, scale, rotation, r, g, b, opacity]
        self.gaussian_params = nn.Parameter(
            torch.randn(num_gaussians, gs_param_dim) * 0.1
        )
    
    def forward(self):
        # 返回形狀 [num_gaussians, gs_param_dim]
        return self.gaussian_params


def render_2dgs(gaussian_params, image_h, image_w):
    """
    2D Gaussian Splatting 渲染 - num_gaussians 個高斯貢獻到整個圖像
    
    Args:
        gaussian_params: [num_gaussians, 8] - 每個高斯點的參數
        image_h: 圖片高度
        image_w: 圖片寬度
    
    Returns:
        rendered_image: [1, 3, H, W]
    """
    device = gaussian_params.device
    num_gaussians = gaussian_params.shape[0]
    
    # 創建輸出圖像
    image = torch.zeros((1, 3, image_h, image_w), device=device)
    
    # 建立座標網格 [H, W]
    yy, xx = torch.meshgrid(
        torch.arange(image_h, device=device, dtype=torch.float32),
        torch.arange(image_w, device=device, dtype=torch.float32),
        indexing="ij"
    )
    
    # 正規化座標到 [0, 1]
    xx_norm = xx / image_w  # [H, W]
    yy_norm = yy / image_h  # [H, W]
    
    # 提取所有高斯參數
    x_center = torch.clamp(gaussian_params[:, 0], 0, 1)  # [H*W]
    y_center = torch.clamp(gaussian_params[:, 1], 0, 1)  # [H*W]
    scale = torch.clamp(gaussian_params[:, 2], 0.001, 0.1)  # [H*W]
    rotation = gaussian_params[:, 3]  # [H*W]
    rgb = torch.clamp(gaussian_params[:, 4:7], 0, 1)  # [H*W, 3]
    opacity = torch.clamp(gaussian_params[:, 7], 0, 1)  # [H*W]
    
    # 計算旋轉矩陣
    cos_rot = torch.cos(rotation)  # [H*W]
    sin_rot = torch.sin(rotation)  # [H*W]
    
    # 逐高斯點累積貢獻
    for g_idx in range(num_gaussians):
        # 計算相對座標
        dx = xx_norm - x_center[g_idx]  # [H, W]
        dy = yy_norm - y_center[g_idx]  # [H, W]
        
        # 應用旋轉
        dx_rot = dx * cos_rot[g_idx] + dy * sin_rot[g_idx]
        dy_rot = -dx * sin_rot[g_idx] + dy * cos_rot[g_idx]
        
        # 計算高斯值
        distance = (dx_rot ** 2 + dy_rot ** 2) / (2 * scale[g_idx] ** 2)
        gaussian_value = torch.exp(-distance) * opacity[g_idx]  # [H, W]
        
        # 累積顏色
        color = rgb[g_idx]  # [3]
        weighted_color = gaussian_value.unsqueeze(-1) * color  # [H, W, 3]
        
        # 使用簡單的累加（不用alpha混合，因為已經有opacity）
        image[0] += weighted_color.permute(2, 0, 1)
    
    return image


def psnr(pred, target):
    """計算 PSNR"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def save_checkpoint(epoch, model, optimizer, path):
    """保存檢查點"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path):
    """載入檢查點"""
    if os.path.exists(path):
        print(f"[Info] Found checkpoint at {path}, loading...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"[Info] Resuming from epoch {start_epoch}")
        return start_epoch
    else:
        print("[Info] No checkpoint found, starting from scratch")
        return 0


# 計算圖片尺寸
num_pixels = image_size * image_size
print(f"Image size: {image_size}x{image_size} = {num_pixels} pixels")
print(f"Number of Gaussians: {num_gaussians}")

# 初始化模型
model = GaussianParameterModel(num_gaussians, gs_param_dim).to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# 檢查是否有檢查點
start_epoch = 0
checkpoint_path = f'./checkpoints/2dgs_full_epoch_0.pth'
if os.path.exists(checkpoint_path):
    print(f"[Info] Found checkpoint at {checkpoint_path}, loading...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"[Info] Resuming from epoch {start_epoch}")
else:
    print("[Info] No checkpoint found, starting from scratch")
    start_epoch = 0

# 數據集 - 直接載入整張圖片
image_dir = "/work/a28133781/datasets/DrealSR_cut"
image_list = [f"DrealSR{str(i).zfill(2)}_LR.png" for i in range(1, 2)]

class FullImageDataset(torch.utils.data.Dataset):
    """直接載入完整圖片的 dataset"""
    def __init__(self, image_dir, image_list, target_size=256):
        self.image_dir = image_dir
        self.image_list = image_list
        self.target_size = target_size
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        
        # 載入 LR 圖片
        lr_path = os.path.join(self.image_dir, img_name)
        lr_image = Image.open(lr_path).convert("RGB")
        
        # 調整大小到 target_size
        lr_image = lr_image.resize((self.target_size, self.target_size), Image.BICUBIC)
        
        # 轉換為 tensor
        lr_tensor = self.to_tensor(lr_image)  # [3, H, W]
        
        # 載入對應的 HR 圖片（用於推理時對比）
        hr_name = img_name.replace("_LR", "_HR")
        hr_path = os.path.join(self.image_dir, hr_name)
        if os.path.exists(hr_path):
            hr_image = Image.open(hr_path).convert("RGB")
            hr_image = hr_image.resize((self.target_size, self.target_size), Image.BICUBIC)
            hr_tensor = self.to_tensor(hr_image)
        else:
            hr_tensor = lr_tensor.clone()
        
        return lr_tensor, hr_tensor

# 創建數據集和數據加載器
dataset = FullImageDataset(
    image_dir=image_dir,
    image_list=image_list,
    target_size=image_size
)

from torch.utils.data import DataLoader
dataloader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True,
    pin_memory=True
)

# 訓練循環
for epoch in tqdm(range(start_epoch, num_epochs)):
    start_time = time.time()
    
    for lr_batch, hr_batch in tqdm(dataloader):
        # lr_batch: [B, 3, H, W]
        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)
        
        # Forward pass：獲取 2DGS 參數
        gaussian_params = model()  # [num_gaussians, 8]
        
        # 2DGS 渲染整張圖像
        rendered_image = render_2dgs(
            gaussian_params=gaussian_params,
            image_h=image_size,
            image_w=image_size
        ).to(device)  # [1, 3, H, W]
        
        # 將 lr_batch 的第一張圖片拿出來比對（因為 batch_size=1）
        rendered_batch = rendered_image.expand(lr_batch.shape[0], -1, -1, -1)
        
        # 保存第一個 epoch 的結果
        if epoch == 0:
            os.makedirs(f"./output/2dgs_full", exist_ok=True)
            save_image(
                lr_batch[0:1],
                f"./output/2dgs_full/lr_epoch{epoch+1}.png"
            )
            save_image(
                rendered_batch[0:1],
                f"./output/2dgs_full/output_epoch{epoch+1}.png"
            )
        
        # 計算損失並反向傳播
        optimizer.zero_grad()
        loss = loss_fn(rendered_batch, lr_batch)
        loss.backward()
        optimizer.step()
    
    # 每 1 個 epoch 進行一次推理
    if (epoch + 1) % 1 == 0:
        with torch.no_grad():
            lr_path = os.path.join(image_dir, "DrealSR01_LR.png")
            
            # 載入完整圖像
            lr_image = Image.open(lr_path).convert("RGB")
            lr_image = lr_image.resize((image_size, image_size), Image.BICUBIC)
            
            to_tensor = transforms.ToTensor()
            lr_tensor = to_tensor(lr_image).to(device)  # [3, H, W]
            
            # 獲取 2DGS 參數並渲染
            gaussian_params = model()  # [num_gaussians, 8]
            rendered_image = render_2dgs(
                gaussian_params=gaussian_params,
                image_h=image_size,
                image_w=image_size
            )  # [1, 3, H, W]
            
            rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
            
            # 計算 PSNR
            epoch_psnr = psnr(rendered_image[0], lr_tensor)
            
            # 保存渲染結果
            os.makedirs(f"./output/2dgs_full", exist_ok=True)
            save_image(
                rendered_image,
                f"./output/2dgs_full/rendered_epoch{epoch+1}.png"
            )
    
    # 每 5 個 epoch 保存檢查點
    if (epoch + 1) % 5 == 0:
        save_checkpoint(
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            path=f'./checkpoints/2dgs_full_epoch_{epoch+1}.pth'
        )
    
    elapsed = time.time() - start_time
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    print(f"Epoch {epoch+1} PSNR on DrealSR01: {epoch_psnr:.2f} | Loss: {loss:.6f} | elapsed time: {h:02d}:{m:02d}:{s:02d}")
