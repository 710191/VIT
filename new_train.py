import torch
import torch.nn as nn
from encoder import Encoder
from render_image import render_image
from PCA import PCA
from CNN import PatchEncoderCNN
from SIREN_MLP import SIREN_MLP
from PIL import Image
from torch.utils.data import DataLoader
from lr_patch_dataset import LRPatchDataset, LRPatchDatasetPreloaded
import numpy as np
import os
from torchvision import transforms
import csv
import time

from tqdm import tqdm

from torchvision.utils import save_image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

hat = Encoder('HAT', 'HAT-L_SRx2_ImageNet-pretrain.pth').to(device)

# 訓練參數
colors = 3 
n = 1000
fft_parameters = 4 
num_epochs = 1000
num_iters = 200
learning_rate = 1e-4
crop_size = 64
patch_size = 64
scale = 1
batch_size = 8  # 一次 GPU 處理 8 個 crop

# save render
save = True

# PCA
in_channels = 64
out_channels = 64
pca = PCA(in_channels, out_channels).to(device)

# CNN
num_downsample = 4
cnn = PatchEncoderCNN(in_channels=out_channels, num_downsample=num_downsample).to(device)

# MLP list
input_dim =  out_channels * ((patch_size * 2 // (2 ** num_downsample)) ** 2)
mlp = SIREN_MLP(input_dim, colors * n * fft_parameters).to(device)

# optimizer
optimizer = torch.optim.Adam(
    list(mlp.parameters()) + list(cnn.parameters()),
    lr=learning_rate
)
loss_fn = nn.MSELoss()

# get start epoch if checkpoint exists
start_epoch = 0  
checkpoint_path = f'./checkpoints/ver3_epoch_{start_epoch}_batch_8.pth'
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
image_list = [f"DrealSR{str(i).zfill(2)}_LR.png" for i in range(1, 2)]


# Create dataset and dataloader ONCE before training loop
dataset = LRPatchDatasetPreloaded(image_dir=image_dir,
                        image_list=image_list,
                        crop_size=crop_size,
                        scale=scale,
                        num_iters=num_iters)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

for epoch in tqdm(range(start_epoch, num_epochs)):
    start_time = time.time()

    iteration = 0
    for lr_batch, hr_batch in tqdm(dataloader):
        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)
        """
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
            lr_batch = lr_tensor.unsqueeze(0).to(device)
            hr_batch = hr_tensor.unsqueeze(0).to(device)
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # encoder forward
        #feat = hat.model.conv_first(lr_batch)
        #lr_latents = hat.model.forward_features(feat) #([8, 180, 64, 64])
        lr_latents = hat(lr_batch) #([8, 64, 128, 128])
        #print("lr_latents", lr_latents.shape)

        # PCA forward
        #latent_pca = pca(lr_latents) 

        # CNN forward
        latent_cnn = cnn(lr_latents) 

        # MLP forward + reshape
        outputs = mlp(latent_cnn)
        outputs = outputs.view(outputs.shape[0], 1, colors, n, fft_parameters)
  


        # render batch 
        rendered_batch = render_image(
            patch_outputs=outputs,
            image_size=(crop_size, crop_size),
            scale=scale,
            patch_size=patch_size
        ).to(device)
        #rendered_batch = torch.clamp(rendered_batch, 0.0, 1.0)

        if iteration == 0:
            save_image(
                lr_batch,
                f"./output/ver4/lr_epoch{epoch+1}.png"
            )
            save_image(
                rendered_batch,
                f"./output/ver4/output_epoch{epoch+1}.png"
            )
        iteration += 1

        # loss + backward
        optimizer.zero_grad()
        loss = loss_fn(rendered_batch, lr_batch)
        loss.backward()
        optimizer.step()
    
    # inference on DrealSR01 every 5 epochs
    if (epoch + 1) % 1 == 0:
        with torch.no_grad():
            lr_path = os.path.join(image_dir, "DrealSR01_LR.png")
            hr_path = os.path.join(image_dir, "DrealSR01_HR.png")
            
            # LR, HR tensor
            lr_image = Image.open(lr_path).convert("RGB")
            hr_image = Image.open(hr_path).convert("RGB")

            # Resize LR Image to be a multiple of crop_size
            lr_width, lr_height = lr_image.size
            new_lr_width = (lr_width // crop_size) * crop_size
            new_lr_height = (lr_height // crop_size) * crop_size
            lr_image = lr_image.resize((new_lr_width, new_lr_height), Image.BICUBIC)
            hr_image = hr_image.resize((new_lr_width * scale, new_lr_height * scale), Image.BICUBIC)

            to_tensor = transforms.ToTensor()
            lr_tensor_full = to_tensor(lr_image).to(device)
            hr_tensor_full = to_tensor(hr_image).to(device)

            # for top, left for cropping 64 * 64
            _, lr_H, lr_W = lr_tensor_full.shape

            render_full = torch.zeros((1, 3, lr_H, lr_W)).to(device)
            for top in range(0, lr_H, crop_size):
                for left in range(0, lr_W, crop_size):
                    if top + crop_size > lr_H or left + crop_size > lr_W:
                        continue
                    
                    lr_tensor = lr_tensor_full[ :, top : top+crop_size, left : left+crop_size]
                    lr_tensor = lr_tensor.unsqueeze(0).to(device)

                    # encoder forward
                    #print("lr_tensor", lr_tensor.shape)
                    #feat = hat.model.conv_first(lr_tensor)
                    #lr_crop_latents = hat.model.forward_features(feat)
                    lr_crop_latents = hat(lr_tensor)

                    # PCA
                    #latent_pca = pca(lr_crop_latents) 

                    # CNN
                    latent_cnn = cnn(lr_crop_latents) 

                    # 丟進MLP後 reshape
                    outputs = mlp(latent_cnn) # [batch, patch_num, colors * n * fft_parameters]
                    outputs = outputs.view(1, 1, colors, n, fft_parameters)  # [batch, patch_num, colors, n, fft_parameters]
            
                    # render reconstructed image
                    rendered_image = render_image(
                        patch_outputs=outputs,
                        image_size=(crop_size, crop_size),
                        scale=scale,
                        patch_size=patch_size
                    ).to(device)
                    rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
                    render_full[ :, :, top : top+crop_size, left : left+crop_size] = rendered_image

            # PSNR
            print("render_full", render_full.shape)
            epoch_psnr = psnr(render_full, lr_tensor_full)
            
            # 存檔查看
            image_np = (render_full[0].permute(1,2,0).detach().cpu().numpy() * 255).astype('uint8')
            img = Image.fromarray(image_np)
            img.save(f"./output/ver4/rendered_epoch{epoch+1}_01.png")

    if (epoch + 1) % 5 == 0:
        save_checkpoint(
            epoch=epoch + 1,
            mlp=mlp,
            cnn=cnn,
            optimizer=optimizer,
            path=f'./checkpoints/ver3_epoch_{epoch+1}_batch_8.pth'
        )

    elapsed = time.time() - start_time
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    print(f"Epoch {epoch+1} PSNR on DrealSR01: {epoch_psnr:.2f} | elapsed time: {h:02d}:{m:02d}:{s:02d}")