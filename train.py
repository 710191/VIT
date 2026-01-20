import torch
import torch.nn as nn
from encoder import Encoder
from render_image import render_image_from_patches
from PCA import PCA
from CNN import PatchEncoderCNN
from MLP import MLP
from PIL import Image
from torch.utils.data import DataLoader
from lr_patch_dataset import LRPatchDataset
import numpy as np
import os
from torchvision import transforms
import csv

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

hat = Encoder('HAT', 'HAT-L_SRx2_ImageNet-pretrain.pth').to(device)

# 訓練參數
colors = 3 
n = 100 
fft_parameters = 4 
num_epochs = 100
num_iters = 10
num_same_crop = 20
learning_rate = 1e-4 * 8
crop_size = 64
patch_size = 64
scale = 1

# save render
save = True

# PCA
in_channels = 180
out_channels = 50
pca = PCA(in_channels, out_channels).to(device)

# CNN
num_downsample = 2
cnn = PatchEncoderCNN(in_channels=out_channels, num_downsample=num_downsample).to(device)

# MLP list
input_dim =  out_channels * ((patch_size // (2 ** num_downsample)) ** 2)
mlp = MLP(input_dim, colors * n * fft_parameters).to(device)

# optimizer
optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# get start epoch if checkpoint exists
start_epoch = 0  
checkpoint_path = f'./checkpoints/ver3_epoch_{start_epoch}.pth'
if os.path.exists(checkpoint_path):
    print(f"[Info] Found checkpoint at {checkpoint_path}, loading...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    mlp.load_state_dict(checkpoint['mlp_state_dict'])
    cnn.load_state_dict(checkpoint['cnn_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"[Info] Resuming from epoch {start_epoch}")
else:
    print("[Info] No checkpoint found, starting from scratch")
    start_epoch = 0

# save checkpoint
def save_checkpoint(epoch, mlp, cnn, optimizer, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'mlp_state_dict': mlp.state_dict(),
        'cnn_state_dict': cnn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


# PSNR 計算函數
def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# training: 02~83
image_dir = "../../dataset/DrealSR_cut64"
image_list = [f"DrealSR{str(i).zfill(2)}_LR.png" for i in range(2, 3)]


for epoch in range(start_epoch, num_epochs):

    # dataset
    dataset = LRPatchDataset(image_dir=image_dir,
                            image_list=image_list,
                            crop_size=crop_size,
                            scale=scale,
                            num_iters=num_iters,
                            num_same_crop=num_same_crop)

    batch_size = 8  # 一次 GPU 處理 8 個 crop
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    for lr_batch, hr_batch in dataloader:
        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)

        # encoder forward
        feat = hat.model.conv_first(lr_batch)
        lr_latents = hat.model.forward_features(feat)

        # PCA forward
        latent_pca = pca(lr_latents) 

        # CNN forward
        latent_cnn = cnn(latent_pca) 

        # MLP forward + reshape
        outputs = mlp(latent_cnn)
        outputs = outputs.view(outputs.shape[0], 1, colors, n, fft_parameters)

        # render batch (仍可逐個 crop render)
        rendered_batch = []
        for b in range(outputs.shape[0]):
            rendered_image = render_image_from_patches(
                patch_outputs=outputs[b:b+1],
                image_size=(crop_size, crop_size),
                batch_idx=0,
                scale=scale,
                patch_size=patch_size
            ).to(device)
            rendered_batch.append(rendered_image.permute(2,0,1))
        rendered_batch = torch.stack(rendered_batch)

        # loss + backward
        optimizer.zero_grad()
        loss = loss_fn(rendered_batch, hr_batch)
        loss.backward()
        optimizer.step()
    
    # inference on DrealSR01 every 5 epochs
    if (epoch + 1) % 1 == 0:
        with torch.no_grad():
            lr_path = os.path.join(image_dir, "DrealSR02_LR.png")
            hr_path = os.path.join(image_dir, "DrealSR02_HR.png")
            
            # LR, HR tensor
            lr_image = Image.open(lr_path).convert("RGB")
            hr_image = Image.open(hr_path).convert("RGB")
            to_tensor = transforms.ToTensor()
            lr_tensor_full = to_tensor(lr_image).to(device)
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

                    # PCA
                    latent_pca = pca(lr_crop_latents) 

                    # CNN
                    latent_cnn = cnn(latent_pca) 

                    # 丟進MLP後 reshape
                    outputs = mlp(latent_cnn) # [batch, patch_num, colors * n * fft_parameters]
                    outputs = outputs.view(1, 1, colors, n, fft_parameters)  # [batch, patch_num, colors, n, fft_parameters]
                    outputs_list.append(outputs)

            # concat回大圖 根據patch_num去塞
            outputs_full_image = torch.cat(outputs_list, dim=1)
            
            # render reconstructed image
            image_size = (lr_H, lr_W)
            rendered_image = render_image_from_patches(
                patch_outputs=outputs_full_image,
                image_size=image_size,
                batch_idx=0,
                scale=scale,
                patch_size=patch_size
            ).to(device)
            rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
            rendered_image_tensor = rendered_image.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            
            # PSNR
            epoch_psnr = psnr(rendered_image.permute(2, 0, 1), lr_tensor_full)
            print(f"Epoch {epoch+1} PSNR on DrealSR01: {epoch_psnr:.2f}")
            
            # 存檔查看
            image_np = (rendered_image.detach().cpu().numpy() * 255).astype('uint8')
            img = Image.fromarray(image_np)
            img.save(f"./output/ver3/rendered_epoch{epoch+1}_01.png")

    if (epoch + 1) % 5 == 0:
        save_checkpoint(
            epoch=epoch + 1,
            mlp=mlp,
            cnn=cnn,
            optimizer=optimizer,
            path=f'./checkpoints/ver3_epoch_{epoch+1}.pth'
        )

    print("epoch", epoch + 1, "finished.")
        
