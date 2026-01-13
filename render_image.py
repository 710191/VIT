import torch
"""
fft: input: 0 ~ 255 tensor

render: input: down
        output: 0 ~ 1 [3, H, W]tensor
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

def render_image_from_patches(
    alphas,             # [3, v_len, u_len]
    phis,               # [3, v_len, u_len]
    v_lists,            # [3, v_len]
    u_lists,            # [3, u_len]
    image_size,         # (H, W) 原始影像大小
    batch_idx=0,
    scale=2,            # 放大倍率
    patch_size = 64     # ViT patch 大小 = 64
):
    """
    回傳:
        recon_patch: [H*scale, W*scale, 3] tensor on device
    """
    H, W = image_size
    Hs, Ws = H * scale, W * scale

    # --------------------------------
    # 3. 取出該 patch 的參數
    # --------------------------------

    # 空間座標
    y, x = torch.meshgrid(
        torch.arange(patch_size*scale, device=device, dtype=torch.float32),
        torch.arange(patch_size*scale, device=device, dtype=torch.float32),
        indexing='ij'
    )

    # 對應回原圖座標（連續座標）
    x = x / scale
    y = y / scale

    H, W, C = patch_size*scale, patch_size*scale, 3
    channels = []

    # 空間座標
    for c in range(3):
        alpha  = alphas[c].to(device)      # [v_len, u_len]
        phi    = phis[c].to(device)        # [v_len, u_len]
        v_list = v_lists[c].to(device)     # [v_len]
        u_list = u_lists[c].to(device)     # [u_len]

        recon_channel = torch.zeros((Hs, Ws), dtype=torch.float32, device=device)

        # ✅ DC component
        recon_channel += alpha[0, 0]

        # ✅ 只跑「一半頻譜」+ ×2
        for u in u_list:
            for v in v_list:
                recon_channel += 2 * alpha[u, v] * torch.cos(
                    2 * torch.pi * (u * y / Hs + v * x / Ws) + phi[u, v]
                )

        # ✅ normalization
        recon_channel /= (len(u_list) * len(v_list) * 2)

        # clip
        recon_channel = torch.clamp(recon_channel, 0, 255) / 255.0

        channels.append(recon_channel)

    img = torch.stack(channels, dim=2).permute(2, 0, 1)  # [3, H*scale, W*scale]
    return img  # tensor, 已經在 device 上
