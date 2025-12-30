import torch
import torch.nn as nn
from encoder import Encoder
from render_image import render_image_from_patches
from MLP import MLP
from PIL import Image
import numpy as np
import os
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

hat = Encoder('HAT', 'HAT-L_SRx2_ImageNet-pretrain.pth').to(device)

# 訓練參數
colors = 3
n = 30
fft_parameters = 4
num_epochs = 100
num_iters = 1
learning_rate = 1e-4
crop_size = 64
scale = 2

# MLP list
patch_size = 64
input_dim = patch_size * patch_size  * 180  # HAT patch latent dimension: H*W*180
mlp = MLP(input_dim, colors * n * fft_parameters).to(device)

# optimizer
optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# PSNR 計算函數
def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# training: 02~83
image_dir = "../../dataset/DrealSR_cut"
image_list = [f"DrealSR{str(i).zfill(2)}_LR.png" for i in range(2, 84)]

for epoch in range(num_epochs):
    for iter in range(num_iters):  # 每個epoch每張圖crop幾次
        for img_name in image_list:
            # LR, HR path
            lr_path = os.path.join(image_dir, img_name)
            hr_name = img_name.replace("_LR", "_HR")
            hr_path = os.path.join(image_dir, hr_name)
            
            # LR, HR tensor
            lr_image = Image.open(lr_path).convert("RGB")
            hr_image = Image.open(hr_path).convert("RGB")
            to_tensor = transforms.ToTensor()
            lr_tensor_full = to_tensor(lr_image)
            hr_tensor_full = to_tensor(hr_image)

            # random top, left for cropping 64 * 64
            _, lr_H, lr_W = lr_tensor_full.shape
            assert lr_H >= crop_size and lr_W >= crop_size
            top = torch.randint(0, lr_H - crop_size + 1, (1,)).item()
            left = torch.randint(0, lr_W - crop_size + 1, (1,)).item()

            # LR, HR crop 有對應位置
            lr_tensor = lr_tensor_full[ :, top : top+crop_size, left : left+crop_size]
            hr_tensor = hr_tensor_full[ :, top * scale:(top + crop_size) * scale, left * scale:(left + crop_size) * scale]

            # 加 batch dimension + device
            lr_tensor = lr_tensor.unsqueeze(0).to(device)
            hr_tensor = hr_tensor.unsqueeze(0).to(device)

            # encoder forward
            #print("lr_tensor", lr_tensor.shape)
            feat = hat.model.conv_first(lr_tensor)
            lr_crop_latents = hat.model.forward_features(feat)

            # 切成 patch 64 * 64
            B, _, _, _ = lr_crop_latents.shape # [1, 180, crop_size, crop_size]
            lr_patches_latents = lr_crop_latents.unfold(3, patch_size, patch_size).unfold(2, patch_size, patch_size) # [1, 180, 2, 2, patch_size, patch_size]
            num_patch_H, num_patch_W = lr_patches_latents.shape[2], lr_patches_latents.shape[3]
            patch_num = num_patch_H * num_patch_W

            # 調整成 MLP 輸入形狀 [B, patch_num, patch_size * patch_size * 180]
            x = lr_patches_latents.permute(0, 2, 3, 4, 5, 1).reshape(B * patch_num, -1)

            # 丟進MLP後 reshape
            outputs = mlp(x) # [batch, patch_num, colors * n * fft_parameters]
            outputs = outputs.view(B, patch_num, colors, n, fft_parameters)  # [batch, patch_num, colors, n, fft_parameters]
            

            # render reconstructed image
            image_size = (64, 64)
            rendered_image = render_image_from_patches(
                patch_outputs=outputs,
                image_size=image_size,
                batch_idx=0,
                scale=2,
                patch_size=patch_size
            ).to(device)
            rendered_image = torch.clamp(rendered_image, 0.0, 1.0) # [H, W, 3]
            rendered_image_tensor = rendered_image.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

            # 計算 loss
            loss = loss_fn(rendered_image_tensor, hr_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # inference on DrealSR01 every 5 epochs
    if (epoch + 1) % 1 == 0:
        with torch.no_grad():
            lr01_path = os.path.join(image_dir, "DrealSR01_LR.png")
            hr01_path = os.path.join(image_dir, "DrealSR01_HR.png")
            
            # open image
            lr_image = Image.open(lr01_path).convert("RGB")
            hr_image = Image.open(hr01_path).convert("RGB")

            # LR, HR tensor
            lr_image = Image.open(lr_path).convert("RGB")
            hr_image = Image.open(hr_path).convert("RGB")
            to_tensor = transforms.ToTensor()
            lr_tensor_full = to_tensor(lr_image)
            hr_tensor_full = to_tensor(hr_image).to(device)

            # for top, left for cropping 64 * 64
            _, lr_H, lr_W = lr_tensor_full.shape

            outputs_list = []
            for top in range(0, lr_H, crop_size):
                for left in range(0, lr_W, crop_size):
                    if top + crop_size > lr_H or left + crop_size > lr_W:
                        continue
                    
                    lr_tensor = lr_tensor_full[ :, top : top+crop_size, left : left+crop_size]
                    lr_tensor = lr_tensor.unsqueeze(0).to(device)

                    # encoder forward
                    #print("lr_tensor", lr_tensor.shape)
                    feat = hat.model.conv_first(lr_tensor)
                    lr_crop_latents = hat.model.forward_features(feat)

                    # 切成 patch 64 * 64
                    B, _, _, _ = lr_crop_latents.shape # [1, 180, crop_size, crop_size]
                    lr_patches_latents = lr_crop_latents.unfold(3, patch_size, patch_size).unfold(2, patch_size, patch_size) # [1, 180, 2, 2, patch_size, patch_size]
                    num_patch_H, num_patch_W = lr_patches_latents.shape[2], lr_patches_latents.shape[3]
                    patch_num = num_patch_H * num_patch_W

                    # 調整成 MLP 輸入形狀 [B, patch_num, patch_size * patch_size * 180]
                    x = lr_patches_latents.permute(0, 2, 3, 4, 5, 1).reshape(B * patch_num, -1)

                    # 丟進MLP後 reshape
                    outputs = mlp(x) # [batch, patch_num, colors * n * fft_parameters]
                    outputs = outputs.view(B, patch_num, colors, n, fft_parameters)  # [batch, patch_num, colors, n, fft_parameters]
                    outputs_list.append(outputs)

            # concat回大圖 根據patch_num去塞
            outputs_full_image = torch.cat(outputs_list, dim=1)
            
            # render reconstructed image
            image_size = (lr_H, lr_W)
            rendered_image = render_image_from_patches(
                patch_outputs=outputs_full_image,
                image_size=image_size,
                batch_idx=0,
                scale=2,
                patch_size=patch_size
            ).to(device)
            rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
            rendered_image_tensor = rendered_image.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            
            # PSNR
            epoch_psnr = psnr(rendered_image.permute(2, 0, 1), hr_tensor_full)
            print(f"Epoch {epoch+1} PSNR on DrealSR01: {epoch_psnr:.2f}")
            
            # 存檔查看
            image_np = (rendered_image.detach().cpu().numpy() * 255).astype('uint8')
            img = Image.fromarray(image_np)
            img.save(f"./output/rendered_epoch{epoch+1}_01.png")
    print("epoch", epoch + 1, "finished.")
        
