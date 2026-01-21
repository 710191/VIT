import torch
import torch.nn as nn
from encoder import Encoder
from render_image import render_image
from MLP import MLP
from CNN import PatchEncoderCNN
from PIL import Image
import numpy as np
import os
from torchvision import transforms
import csv
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

hat = Encoder('HAT', 'HAT-L_SRx2_ImageNet-pretrain.pth').to(device)

# 訓練參數
colors = 3 
n = 100 
fft_parameters = 4 
num_epochs = 1000
num_iters = 10
num_same_crop = 20
learning_rate = 1e-4
crop_size = 64
patch_size = 64
scale = 1

# save render
save = True

# MLP list
input_dim = 2880
mlp = MLP(input_dim, colors * n * fft_parameters).to(device)

# CNN
cnn = PatchEncoderCNN(in_channels=180).cuda()

# optimizer
optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# get start epoch if checkpoint exists
start_epoch = 100  
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

# test
def save_latent_cnn_n_by_param(latent_cnn, n, fft_parameters, filename='latent_cnn.csv', folder='csv_output'):
    """
    latent_cnn: [1, 1, n*fft_parameters] tensor
    reshape 成 [n, fft_parameters] 並存 CSV
    """
    os.makedirs(folder, exist_ok=True)

    # 去掉 batch/channel 維度
    latent_np = latent_cnn.squeeze().cpu().numpy()  # shape [n*fft_parameters]

    # reshape 成 [n, fft_parameters]
    latent_np = latent_np.reshape(n, fft_parameters)

    path = os.path.join(folder, filename)

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in latent_np:
            writer.writerow(row)

    print(f"latent_cnn saved to {path} with shape [{n}, {fft_parameters}]")

def save_hwc3_tensor_to_csv(tensor, path):
    """
    tensor: [H, W, 3]
    CSV: H rows, W cols, each cell = "R G B"
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    t = tensor.detach().cpu()
    H, W, C = t.shape
    assert C == 3

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for h in range(H):
            row = []
            for w in range(W):
                r, g, b = t[h, w].tolist()
                row.append(f"{r} {g} {b}")
            writer.writerow(row)

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
    start_time = time.time()
    for img_name in image_list:
        for iter in range(num_iters):  # 每個epoch每張圖crop幾次

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

            for same_crop in range(num_same_crop):  # 每個crop跑幾次
                # encoder forward
                #print("lr_tensor", lr_tensor.shape)
                feat = hat.model.conv_first(lr_tensor)
                lr_crop_latents = hat.model.forward_features(feat)
                
                # CNN: [1, 180, crop_size, crop_size] -> [1, 1, n*fft_parameters]
                latent_cnn = cnn(lr_crop_latents) 

                # 丟進MLP後 reshape
                outputs = mlp(latent_cnn) # [batch, patch_num, colors * n * fft_parameters]
                outputs = outputs.view(1, 1, colors, n, fft_parameters)  # [batch, patch_num, colors, n, fft_parameters]

                # render reconstructed image
                image_size = (64, 64)
                rendered_image = render_image(
                    patch_outputs=outputs,
                    image_size=image_size,
                    scale=scale,
                    patch_size=patch_size
                ).to(device)
                rendered_image = torch.clamp(rendered_image, 0.0, 1.0) # [H, W, 3]
                rendered_image_tensor = rendered_image.permute(0, 3, 1, 2)  # [1, 3, H, W]

                # 計算 loss
                optimizer.zero_grad()
                loss = loss_fn(rendered_image_tensor, lr_tensor)

                loss.backward()
                optimizer.step()
    
    # inference on DrealSR01 every 5 epochs
    if (epoch + 1) % 1 == 0:
        with torch.no_grad():
            lr01_path = os.path.join(image_dir, "DrealSR02_LR.png")
            hr01_path = os.path.join(image_dir, "DrealSR02_HR.png")
            
            # open image
            lr_image = Image.open(lr01_path).convert("RGB")
            hr_image = Image.open(hr01_path).convert("RGB")

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

                    # CNN: [1, 180, crop_size, crop_size] -> [1, 1, n*fft_parameters]
                    latent_cnn = cnn(lr_crop_latents) 

                    # 丟進MLP後 reshape
                    outputs = mlp(latent_cnn) # [batch, patch_num, colors * n * fft_parameters]
                    outputs = outputs.view(1, 1, colors, n, fft_parameters)  # [batch, patch_num, colors, n, fft_parameters]
                    outputs_list.append(outputs)

            # concat回大圖 根據patch_num去塞
            outputs_full_image = torch.cat(outputs_list, dim=1)
            
            # render reconstructed image
            image_size = (lr_H, lr_W)
            rendered_image = render_image(
                patch_outputs=outputs_full_image,
                image_size=image_size,
                scale=scale,
                patch_size=patch_size
            ).to(device)
            rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
            rendered_image_tensor = rendered_image.permute(0, 3, 1, 2)  # [1, 3, H, W]
            
            # PSNR
            epoch_psnr = psnr(rendered_image.squeeze(0).permute(2, 0, 1), lr_tensor_full)
            
            # 存檔查看
            image_np = (rendered_image.squeeze(0).detach().cpu().numpy() * 255).astype('uint8')
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

    elapsed = time.time() - start_time
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    print(f"Epoch {epoch+1} PSNR on DrealSR01: {epoch_psnr:.2f} | elapsed time: {h:02d}:{m:02d}:{s:02d}")
        
