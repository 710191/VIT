import torch

def render_image_from_patches(
    patch_outputs,      # [batch, patch_num, 3, n, 4]
    image_size,         # (H, W) 原始影像大小
    batch_idx=0,
    scale=2,            # 放大倍率
    patch_size = 32     # ViT patch 大小 = 32
):
    """
    回傳:
        image: [H*scale, W*scale, 3]
    """
    device = patch_outputs.device
    H, W = image_size
    Hs, Ws = H * scale, W * scale

    # --------------------------------
    # 1. 建立 pixel 座標 (放大後)
    # --------------------------------
    y, x = torch.meshgrid(
        torch.arange(Hs, device=device),
        torch.arange(Ws, device=device),
        indexing="ij"
    )

    # 對應回原圖座標（連續座標）
    x = x.float() / scale
    y = y.float() / scale

    # --------------------------------
    # 2. 計算 ViT patch index
    # --------------------------------
    patch_cols = W // patch_size
    patch_rows = H // patch_size

    patch_row = torch.clamp((y // patch_size).long(), max=patch_rows - 1)
    patch_col = torch.clamp((x // patch_size).long(), max=patch_cols - 1)

    patch_idx = patch_row * patch_cols + patch_col   # [Hs, Ws]

    # --------------------------------
    # 3. 取出該 patch 的參數
    # --------------------------------
    # [Hs, Ws, 3, n, 4]
    params = patch_outputs[batch_idx][patch_idx]

    alpha   = params[..., 0]   # [Hs, Ws, 3, n]
    phi     = params[..., 1]
    omega_x = params[..., 2]
    omega_y = params[..., 3]

    # --------------------------------
    # 4. 計算 V(x, y)
    # --------------------------------
    x = x.view(Hs, Ws, 1, 1)
    y = y.view(Hs, Ws, 1, 1)

    value = alpha * torch.sin(
        omega_x * x + omega_y * y + phi
    )

    image = value.sum(dim=-1)   # sum over n → [Hs, Ws, 3]

    return image
