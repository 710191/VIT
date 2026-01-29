import torch
import torch.nn as nn
from encoder import Encoder
from render_image import render_image_from_patches
from MLP import MLP
from CNN import PatchEncoderCNN
from PIL import Image
import numpy as np
import os
from torchvision import transforms
import csv

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

hat = Encoder('HAT', 'HAT-L_SRx2_ImageNet-pretrain.pth').to(device)
to_tensor = transforms.ToTensor()

# 訓練參數
colors = 3 
n = 32  # u_len = 2n, v_len = n
fft_parameters = 4 
num_epochs = 10000
num_iters = 100
num_same_crop = 20
learning_rate = 1e-2
crop_size = 64
patch_size = 64
scale = 1

# save render
save = True

# MLP list
input_dim = 11520 // 4
mlp = MLP(input_dim, colors * (n * n * 2 * 2)).to(device)

# CNN
cnn = PatchEncoderCNN(in_channels=180, num_downsample=4).cuda()

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
image_dir = "../../dataset/DrealSR_cut128"
image_list = [f"DrealSR{str(i).zfill(2)}_LR.png" for i in range(2, 3)]

for epoch in range(num_epochs):

    optimizer.zero_grad()
    loss_parameters = 0.0

    for img_name in image_list:
        lr_tensor_list = []
        hr_tensor_list = []
        rendered_image_tensor_list = []

        for iter in range(num_iters):  # 每個epoch每張圖crop幾次
            # LR, HR path
            lr_path = os.path.join(image_dir, img_name)
            hr_name = img_name.replace("_LR", "_HR")
            hr_path = os.path.join(image_dir, hr_name)
            
            # open image
            lr_image = Image.open(lr_path).convert("RGB")
            hr_image = Image.open(hr_path).convert("RGB")

            # LR, HR tensor
            lr_tensor_full = to_tensor(lr_image).to(device)  # [3, H, W],  0 ~ 1 
            hr_tensor_full = to_tensor(hr_image).to(device)  # [3, H, W],  0 ~ 1

            _, lr_H, lr_W = lr_tensor_full.shape

            # random top, left for cropping 64 * 64
            top = 0 #torch.randint(0, lr_H - crop_size + 1, (1,)).item()
            left = 0 #torch.randint(0, lr_W - crop_size + 1, (1,)).item()

            # LR, HR crop 有對應位置
            lr_tensor = lr_tensor_full[ :, top : top+crop_size, left : left+crop_size]
            hr_tensor = hr_tensor_full[ :, top * scale:(top + crop_size) * scale, left * scale:(left + crop_size) * scale]

            # 加 batch dimension + device
            lr_tensor = lr_tensor.unsqueeze(0).to(device)
            hr_tensor = hr_tensor.unsqueeze(0).to(device)

            lr_tensor_list.append(lr_tensor)
            hr_tensor_list.append(hr_tensor)
            
            with torch.no_grad():
                feat = hat.model.conv_first(lr_tensor)
                hat_output = hat.model.forward_features(feat)

            # CNN: [1, 180, crop_size, crop_size] -> [1, 1, n*fft_parameters]
            lr_tensor_cnn = cnn(hat_output)
            latent_cnn = lr_tensor_cnn.reshape(1, 1, -1)


            # 丟進MLP後 reshape
            outputs = mlp(latent_cnn) # [batch, patch_num, colors * n * fft_parameters]
            outputs = outputs.view(2, 3, 2 * n, n)  # [batch, patch_num, colors, n, fft_parameters]

            # PyTorch FFT
            F = torch.fft.fft2(lr_tensor.squeeze(0) * 255, dim=(1,2)) 
            alphas = torch.abs(F)   # [C, H, W]
            phis   = torch.angle(F) # [C, H, W]
            u_lists = torch.arange(2*n, device=device).squeeze(0).repeat(3, 1)
            v_lists = torch.arange(1, n, device=device).squeeze(0).repeat(3, 1)
            
            alphas_half = alphas[:, :2 * n, :n]
            phis_half = phis[:, :2 * n, :n]

            # render reconstructed image
            image_size = (64, 64)
            rendered_image = render_image_from_patches(
                alphas = alphas_half, # + outputs[0, :, :, :], # alphas
                phis = phis_half, # + outputs[1, :, :, :],   # phis
                v_lists = v_lists,
                u_lists = u_lists,
                image_size=image_size,
                batch_idx=0,
                scale=scale,
                patch_size=patch_size
            ).to(device)
            rendered_image_tensor = rendered_image.unsqueeze(0)  # [1, 3, H, W]
            rendered_image_tensor_list.append(rendered_image_tensor)

            # parameter loss
            loss_parameter = loss_fn(outputs[0], alphas_half) * 0.000000001 + loss_fn(outputs[1], phis_half)*0.1
            loss_parameters += loss_parameter

    # parameter loss average
    avg_loss_parameter = loss_parameters / (len(image_list))

    # all loss: img loss + parameter loss
    rendered_image_tensor_batch = torch.cat(rendered_image_tensor_list, dim=0)
    lr_tensor_batch = torch.cat(lr_tensor_list, dim=0)
    hr_tensor_batch = torch.cat(hr_tensor_list, dim=0)
    loss = loss_fn(rendered_image_tensor_batch, lr_tensor_batch) #+ avg_loss_parameter
    print(f"Epoch {epoch+1}, Avg Parameter Loss: {avg_loss_parameter.item():.6f}")
    print("img loss:", loss_fn(rendered_image_tensor_batch, lr_tensor_batch).item())
    print("loss:", loss.item())

    #loss.backward()
    #optimizer.step()
    
    # inference on DrealSR01 every 5 epochs
    if (epoch + 1) % 1 == 0:
        with torch.no_grad():
            lr_path = os.path.join(image_dir, "DrealSR02_LR.png")
            hr_path = os.path.join(image_dir, "DrealSR02_HR.png")
            
            # open image
            lr_image = Image.open(lr_path).convert("RGB")
            hr_image = Image.open(hr_path).convert("RGB")

            # LR, HR tensor
            lr_tensor_full = to_tensor(lr_image).to(device)  # [3, H, W],  0 ~ 1 
            hr_tensor_full = to_tensor(hr_image).to(device)  # [3, H, W],  0 ~ 1

            # for top, left for cropping 64 * 64
            _, lr_H, lr_W = lr_tensor_full.shape

            predict_img = torch.zeros(
                (3, lr_H*scale, lr_W*scale),
                dtype=torch.float32,
                device=device
            )

            for top in range(0, lr_H, crop_size):
                for left in range(0, lr_W, crop_size):

                    if top + crop_size > lr_H or left + crop_size > lr_W:
                        continue
                    
                    lr_tensor = lr_tensor_full[:, top : top+crop_size, left : left+crop_size]
                    
                    feat = hat.model.conv_first(lr_tensor.unsqueeze(0))
                    hat_output = hat.model.forward_features(feat)
                    
                    # CNN: [1, 180, crop_size, crop_size] -> [1, 1, n*fft_parameters]
                    lr_tensor_cnn = cnn(hat_output)
                    latent_cnn = lr_tensor_cnn.reshape(1, 1, -1)

                    # 丟進MLP後 reshape
                    outputs = mlp(latent_cnn) # [batch, patch_num, colors * n * fft_parameters]
                    outputs = outputs.view(2, 3, 2 * n, n)  # [batch, patch_num, colors, n, fft_parameters]
                            
                    # PyTorch FFT
                    F = torch.fft.fft2(lr_tensor * 255, dim=(1,2)) 
                    alphas = torch.abs(F)   # [C, H, W]
                    phis   = torch.angle(F) # [C, H, W]
                    u_lists = torch.arange(n*2, device=device).squeeze(0).repeat(3, 1)
                    v_lists = torch.arange(1, n, device=device).squeeze(0).repeat(3, 1)

                    alphas_half = alphas[:, :2 * n, :n]
                    phis_half = phis[:, :2 * n, :n]

                    # render reconstructed image
                    image_size = (patch_size, patch_size)
                    rendered_image = render_image_from_patches(
                        alphas = alphas_half + outputs[0, :, :, :], # alphas
                        phis = phis_half + outputs[1, :, :, :],   # phis
                        v_lists = v_lists,
                        u_lists = u_lists,
                        image_size=image_size,
                        batch_idx=0,
                        scale=scale,
                        patch_size=patch_size
                    ) 

                    top_sr  = top * scale
                    left_sr = left * scale

                    predict_img[
                        :,
                        top_sr : top_sr + patch_size * scale,
                        left_sr : left_sr + patch_size * scale
                    ] = rendered_image
                    
            # PSNR
            epoch_psnr = psnr(predict_img, lr_tensor_full)
            print(f"Epoch {epoch+1} PSNR on DrealSR01: {epoch_psnr:.2f}")
            
            # 5️⃣ 存檔
            predict_img = predict_img.permute(1, 2, 0) * 255  # [H, W, 3] 0 ~ 255
            predict_img = predict_img.detach().cpu().numpy().astype(np.uint8)
            recon_img = Image.fromarray(predict_img)
            recon_img.save(f"./output/render/rendered_epoch{epoch+1}_01.png")

    print("epoch", epoch + 1, "finished.")
        
