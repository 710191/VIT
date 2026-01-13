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

#print("new")

# 訓練參數
colors = 3 
n = 32  # u_len = v_len = n
fft_parameters = 4 
num_epochs = 10000
num_iters = 10
num_same_crop = 20
learning_rate = 1e-4
crop_size = 64
patch_size = 64
scale = 1

# save render
save = True

# MLP list
input_dim = 192
mlp = MLP(input_dim, colors * (n * n * 2 * 2)).to(device)

# CNN
cnn = PatchEncoderCNN(in_channels=3, num_downsample=3).cuda()

# optimizer
optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

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
        rendered_image_tensor_list = []

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
            top = 0 #torch.randint(0, lr_H - crop_size + 1, (1,)).item()
            left = 0 #torch.randint(0, lr_W - crop_size + 1, (1,)).item()

            #top = top - top % crop_size
            #left = left - left % crop_size

            # LR, HR crop 有對應位置
            lr_tensor = lr_tensor_full[ :, top : top+crop_size, left : left+crop_size]
            hr_tensor = hr_tensor_full[ :, top * scale:(top + crop_size) * scale, left * scale:(left + crop_size) * scale]

            # 加 batch dimension + device
            lr_tensor = lr_tensor.unsqueeze(0).to(device)
            hr_tensor = hr_tensor.unsqueeze(0).to(device)

            lr_tensor_list.append(lr_tensor)
            
            # CNN: [1, 180, crop_size, crop_size] -> [1, 1, n*fft_parameters]
            #print(lr_tensor.shape)
            lr_tensor_cnn = cnn(lr_tensor)
            latent_cnn = lr_tensor_cnn.reshape(1, 1, -1)

            # 丟進MLP後 reshape
            outputs = mlp(latent_cnn) # [batch, patch_num, colors * n * fft_parameters]
            outputs = outputs.view(2, 3, 2 * n, n)  # [batch, patch_num, colors, n, fft_parameters]

            #num_zeros = (outputs == 0).sum()
            #total = outputs.numel()  # tensor 總元素數
            #print(f"zeros: {num_zeros.item()}, ratio: {num_zeros.item()/total:.4f}")

            #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~\noutputs shape:", outputs.shape)
            #print("outputs", outputs)
            #print(outputs[0, :, :, :].sum())

            # 2️⃣ 使用 PyTorch FFT
            F = torch.fft.fft2(lr_tensor, dim=(0,1))  # [H, W, 3], complex
            F = F.squeeze(0)                   # [C, H, W]

            alphas = torch.abs(F)   # [C, H, W]
            phis   = torch.angle(F) # [C, H, W]
            u_lists = torch.arange(64, device=device).squeeze(0).repeat(3, 1)
            v_lists = torch.arange(1, 64//2, device=device).squeeze(0).repeat(3, 1)

            # render reconstructed image
            image_size = (64, 64)
            rendered_image = render_image_from_patches(
                alphas = outputs[0, :, :, :], # alphas
                phis = outputs[1, :, :, :],   # phis
                v_lists = v_lists,
                u_lists = u_lists,
                image_size=image_size,
                batch_idx=0,
                scale=scale,
                patch_size=patch_size
            ).to(device)
            rendered_image = torch.clamp(rendered_image, 0.0, 1.0) # [H, W, 3]
            rendered_image_tensor = rendered_image.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            rendered_image_tensor_list.append(rendered_image_tensor)

            alphas_half = alphas[:, :2 * n, :n]
            phis_half = phis[:, :2 * n, :n]
            loss_parameter = loss_fn(outputs[0], alphas_half) + loss_fn(outputs[1], phis_half)
            loss_parameters += loss_parameter

    avg_loss_parameter = loss_parameters / (len(image_list) * num_iters)
    print(f"Epoch {epoch+1}, Avg Parameter Loss: {avg_loss_parameter.item():.6f}")

    rendered_image_tensor_batch = torch.cat(rendered_image_tensor_list, dim=0)
    lr_tensor_batch = torch.cat(lr_tensor_list, dim=0)
    # 計算 loss
    loss = loss_fn(rendered_image_tensor_batch, lr_tensor_batch) + avg_loss_parameter
    print("loss:", loss.item())

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
            lr_img_np = np.array(lr_image).astype(np.float32)        # [H, W, 3], float32
            lr_tensor_full = torch.from_numpy(lr_img_np).to(device)     # tensor 在 GPU
            hr_img_np = np.array(hr_image).astype(np.float32)        # [H, W, 3], float32
            hr_tensor_full = torch.from_numpy(hr_img_np).to(device)     # tensor 在 GPU

            print(lr_tensor_full.shape)

            # for top, left for cropping 64 * 64
            lr_H, lr_W, _ = lr_tensor_full.shape

            predict_img = torch.zeros(
                (3, lr_H*scale, lr_W*scale),
                dtype=torch.float32,
                device=device
            )

            for top in range(0, lr_H, crop_size):
                for left in range(0, lr_W, crop_size):
                    if top + crop_size > lr_H or left + crop_size > lr_W:
                        continue
                    
                    lr_tensor = lr_tensor_full[top : top+crop_size, left : left+crop_size, :]

                    
                    """
                    #print("lr_tensor", lr_tensor.shape)
                    lr_tensor_cnn = cnn(lr_tensor.permute(2,0,1))
                    latent_cnn = lr_tensor_cnn.reshape(1, 1, -1)

                    # 丟進MLP後 reshape
                    outputs = mlp(latent_cnn) # [batch, patch_num, colors * n * fft_parameters]
                    outputs = outputs.view(2, 3, 2 * n, n)  # [batch, patch_num, colors, n, fft_parameters]
                    """
                            
                    # 2️⃣ 使用 PyTorch FFT
                    #print("lr_tensor.shape", lr_tensor.shape)
                    F = torch.fft.fft2(lr_tensor, dim=(0,1))  # [H, W, 3], complex
                    F = F.permute(2, 0, 1)                 # [C, H, W]

                    alphas = torch.abs(F)   # [C, H, W]
                    phis   = torch.angle(F) # [C, H, W]
                    u_lists = torch.arange(64, device=device).squeeze(0).repeat(3, 1)
                    v_lists = torch.arange(1, 64//2, device=device).squeeze(0).repeat(3, 1)

                    # render reconstructed image
                    image_size = (patch_size, patch_size)
                    rendered_image = render_image_from_patches(
                        alphas = alphas,
                        phis = phis,
                        v_lists = v_lists,
                        u_lists = u_lists,
                        image_size=image_size,
                        batch_idx=0,
                        scale=scale,
                        patch_size=patch_size
                    ).permute(2, 0, 1)  # [3, H, W]
                   # print("rendered_image.shape", rendered_image.shape)    

                    top_sr  = top * scale
                    left_sr = left * scale

                    predict_img[
                        :,
                        top_sr : top_sr + patch_size * scale,
                        left_sr : left_sr + patch_size * scale
                    ] = rendered_image
                    
            # PSNR
            #print(predict_img.shape, lr_tensor_full.permute(2,0,1).shape)
            #print("predict_img", predict_img)
            #print("lr_tensor_full", lr_tensor_full.permute(2,0,1))

            predict_img_norm = predict_img / 255.0
            lr_tensor_norm = lr_tensor_full.permute(2,0,1) / 255.0  # 如果你的PSNR函數用 CHW
            epoch_psnr = psnr(predict_img_norm, lr_tensor_norm)
            print(f"Epoch {epoch+1} PSNR on DrealSR01: {epoch_psnr:.2f}")
            
            # 5️⃣ 存檔
            predict_img = predict_img.permute(1, 2, 0)  # [H, W, 3]
            predict_img = predict_img.detach().cpu().numpy().astype(np.uint8)
            recon_img = Image.fromarray(predict_img)
            recon_img.save(f"./output/render/rendered_epoch{epoch+1}_01.png")

    print("epoch", epoch + 1, "finished.")
        
