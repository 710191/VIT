import os
import torch
import numpy as np
from PIL import Image
from render_image import render_image_from_patches  # 你的重建函數

device = "cuda" if torch.cuda.is_available() else "cpu"

image_dir = "../../dataset/DrealSR_cut64"
image_list = [f"DrealSR{str(i).zfill(2)}_LR.png" for i in range(2, 3)]

recon_dir = "./output/test_fft_recon"
os.makedirs(recon_dir, exist_ok=True)

for img_name in image_list:
    # 1️⃣ 讀圖
    img_path = os.path.join(image_dir, img_name)
    img_pil = Image.open(img_path).convert("RGB")

    # ✅ 直接轉成 numpy，再轉 tensor 並放 GPU
    img_np = np.array(img_pil).astype(np.float32)        # [H, W, 3], float32
    img_tensor = torch.from_numpy(img_np).to(device)     # tensor 在 GPU

    H, W, C = img_tensor.shape

    # 2️⃣ 使用 PyTorch FFT
    F = torch.fft.fft2(img_tensor, dim=(0,1))  # [H, W, 3], complex
    F = F.permute(2, 0, 1)                     # [C, H, W]

    alphas = torch.abs(F)   # [C, H, W]
    phis   = torch.angle(F) # [C, H, W]

    # 3️⃣ 頻率索引
    u_lists = torch.arange(H, device=device).unsqueeze(0).repeat(C, 1)       # [C, H]
    v_lists = torch.arange(1, W//2, device=device).unsqueeze(0).repeat(C, 1) # [C, W//2-1]

    # 4️⃣ 呼叫重建
    img_recon = render_image_from_patches(
        alphas,
        phis,
        v_lists,
        u_lists,
        image_size=(H,W),
        batch_idx=0,
        scale=1,
    )

    # 5️⃣ 存檔
    img_recon = img_recon.detach().cpu().numpy().astype(np.uint8)
    recon_img = Image.fromarray(img_recon)
    recon_path = os.path.join(recon_dir, f"recon_{img_name}")
    recon_img.save(recon_path)

    print(f"重建圖片已存: {recon_path}")
