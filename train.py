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
channels = 3
n = 30
params_per_group = 4
num_epochs = 100
learning_rate = 1e-4

# MLP list
input_dim = 32*32*180  # CLIP patch latent dimension
mlp = MLP(input_dim, 3 * n * 4).to(device)

# optimizer
optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()  # 用 HR 當 target

# 圖片路徑列表 02~83
image_dir = "../../dataset/DrealSR_cut"
image_list = [f"DrealSR{str(i).zfill(2)}_LR.png" for i in range(2, 3)]

# PSNR 計算函數
def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

for epoch in range(num_epochs):
    for reapeat in range(10):  # 每張圖多訓練幾次
        for img_name in image_list:
            # LR 路徑
            lr_path = os.path.join(image_dir, img_name)
            # HR 路徑
            hr_name = img_name.replace("_LR", "_HR")
            hr_path = os.path.join(image_dir, hr_name)
            
            # encoder
            lr_image = Image.open(lr_path).convert("RGB")
            hr_image = Image.open(hr_path).convert("RGB")

            # 轉 tensor（先不要 unsqueeze）
            to_tensor = transforms.ToTensor()
            lr_tensor_full = to_tensor(lr_image)
            hr_tensor_full = to_tensor(hr_image)

            # =======================
            # random crop
            # =======================
            _, H, W = lr_tensor_full.shape
            crop_size = 64
            scale = 2

            # 確保可 crop
            assert H >= crop_size and W >= crop_size

            top = torch.randint(0, H - crop_size + 1, (1,)).item()
            left = torch.randint(0, W - crop_size + 1, (1,)).item()

            # LR crop
            lr_tensor = lr_tensor_full[
                :,
                top:top + crop_size,
                left:left + crop_size
            ]

            # HR crop（對應位置）
            hr_tensor = hr_tensor_full[
                :,
                top * scale:(top + crop_size) * scale,
                left * scale:(left + crop_size) * scale
            ]

            # 加 batch dimension + device
            lr_tensor = lr_tensor.unsqueeze(0).to(device)
            hr_tensor = hr_tensor.unsqueeze(0).to(device)

            # forward
            #print("lr_tensor", lr_tensor.shape)
            feat = hat.model.conv_first(lr_tensor)
            patch_latents_only = hat.model.forward_features(feat)

            B, C, H, W = patch_latents_only.shape
            patch_size = 32

            # 先切 patch
            patches = patch_latents_only.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            num_patch_H, num_patch_W = patches.shape[2], patches.shape[3]
            patch_num = num_patch_H * num_patch_W

            # 維持 x 這個名字，把每個 patch 展平成 MLP 輸入
            x = patches.permute(0, 2, 3, 1, 4, 5).reshape(B * patch_num, -1)
            #print("B, patch_num", B, patch_num)
            #print("x", x.shape)


            # concat 後還原 shape
            outputs = mlp(x)
            outputs = outputs.view(B, patch_num, 3, n, 4)  # [batch, patch_num, total_params]
            #print("outputs", outputs.shape)
            
            # render reconstructed image
            image_size = (64, 64)
            rendered_image = render_image_from_patches(
                patch_outputs=outputs,
                image_size=image_size,
                batch_idx=0,
                scale=2
            ).to(device)
            rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
            rendered_image_tensor = rendered_image.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

            #print(rendered_image_tensor.shape, hr_tensor.shape, sep =" | ")

            # 計算 loss
            loss = loss_fn(rendered_image_tensor, hr_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    #23131231
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            lr01_path = os.path.join(image_dir, "DrealSR01_LR.png")
            hr01_path = os.path.join(image_dir, "DrealSR01_HR.png")
            
            # open image
            lr_image = Image.open(lr01_path).convert("RGB")
            hr_image = Image.open(hr01_path).convert("RGB")

            # 轉 tensor（先不要 unsqueeze）
            to_tensor = transforms.ToTensor()
            lr_tensor_full = to_tensor(lr_image).to(device)
            hr_tensor_full = to_tensor(hr_image).to(device)
            #print("lr_tensor_full", lr_tensor_full.shape)

            # =======================
            # random crop
            # =======================
            _, H, W = lr_tensor_full.shape
            #print("H, W", H, W)
            crop_size = 32
            scale = 2

            outputs_list = []
            for top in range(0, H, crop_size):
                for left in range(0, W, crop_size):
                    if top + crop_size > H or left + crop_size > W:
                        continue
                    
                    lr_tensor = lr_tensor_full[
                        :,
                        top:top + crop_size,
                        left:left + crop_size
                    ]
                    lr_tensor = lr_tensor.unsqueeze(0).to(device)

                    # forward
                    #print("lr_tensor", lr_tensor.shape)
                    feat = hat.model.conv_first(lr_tensor)
                    patch_latents_only = hat.model.forward_features(feat)

                    B, _, _, _ = patch_latents_only.shape
                    patch_size = 32

                    # 先切 patch
                    patches = patch_latents_only.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
                    num_patch_H, num_patch_W = patches.shape[2], patches.shape[3]
                    patch_num = num_patch_H * num_patch_W

                    # 維持 x 這個名字，把每個 patch 展平成 MLP 輸入
                    x = patches.permute(0, 2, 3, 1, 4, 5).reshape(B * patch_num, -1)
                    #print("B, patch_num", B, patch_num)
                    #print("x", x.shape)


                    # concat 後還原 shape
                    outputs = mlp(x)
                    outputs = outputs.view(B, 1, 3, n, 4)  # [batch, patch_num, total_params]
                    outputs_list.append(outputs)
                    #print("top, left", top, left, outputs.shape)
            outputs_full_image = torch.cat(outputs_list, dim=1)
            #print("outputs_full_image", outputs_full_image.shape)
            
            # render reconstructed image
            
            image_size = (H, W)
            #print("image_size", image_size)
            rendered_image = render_image_from_patches(
                patch_outputs=outputs_full_image,
                image_size=image_size,
                batch_idx=0,
                scale=2
            ).to(device)
            rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
            rendered_image_tensor = rendered_image.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

            pred = rendered_image.permute(2, 0, 1)
            target = hr_tensor_full
            #print(torch.isnan(pred).any(), torch.isinf(pred).any())
            #print(torch.isnan(target).any(), torch.isinf(target).any())
            
            
            # PSNR
            #print("rendered_image_tensor", rendered_image_tensor.shape)
            #print("hr_tensor_full", hr_tensor_full.shape)
            epoch_psnr = psnr(rendered_image.permute(2, 0, 1), hr_tensor_full)
            print(f"Epoch {epoch+1} PSNR on DrealSR01: {epoch_psnr:.2f}")
            
            
            # 存檔查看
            image_np = (rendered_image.detach().cpu().numpy() * 255).astype('uint8')
            img = Image.fromarray(image_np)
            img.save(f"./output/rendered_epoch{epoch+1}_01.png")
    print("epoch", epoch + 1)
        
