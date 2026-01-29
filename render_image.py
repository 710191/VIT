import torch

def render_image(
    patch_outputs,      # [B, patch_num, 3, n, 4]
    image_size,         # (H, W)
    scale=2,
    patch_size=64
):
    """
    Vectorized version: 返回 [B, H*scale, W*scale, 3]
    """
    device = patch_outputs.device
    B, patch_num, C, n, _ = patch_outputs.shape
    H, W = image_size
    Hs, Ws = H * scale, W * scale

    # 1. 建立座標網格
    y, x = torch.meshgrid(
        torch.arange(Hs, device=device),
        torch.arange(Ws, device=device),
        indexing="ij"
    )
    x = x.float() / scale
    y = y.float() / scale

    # 2. 計算 patch index
    patch_rows, patch_cols = H // patch_size, W // patch_size
    patch_row = torch.clamp((y // patch_size).long(), max=patch_rows - 1)
    patch_col = torch.clamp((x // patch_size).long(), max=patch_cols - 1)
    patch_idx = patch_row * patch_cols + patch_col  # [Hs, Ws]

    x_local = x - patch_col * patch_size
    y_local = y - patch_row * patch_size

    x_local /= patch_size
    y_local /= patch_size

    # 3. 擴展 batch
    patch_idx_flat = patch_idx.view(1, -1).expand(B, Hs*Ws)  # [B, Hs*Ws]

    # flatten patch dimension以方便 gather
    #patch_outputs_flat = patch_outputs.permute(0, 1, 2, 4, 3).reshape(B, patch_num, C*4*n)  # [B, patch_num, C*4*n]
    patch_outputs_flat = patch_outputs.reshape(B, patch_num, C*n*4)  # [B, patch_num, C*n*4]

    # gather對應 patch
    gathered = torch.gather(patch_outputs_flat, 1, patch_idx_flat.unsqueeze(-1).expand(-1, -1, C*4*n))  # [B, Hs*Ws, C*4*n]
    gathered = gathered.view(B, Hs, Ws, C, n, 4)  # [B, Hs, Ws, C, n, 4]

    alpha   = gathered[..., 0]
    phi     = gathered[..., 1]
    omega_x = gathered[..., 2]
    omega_y = gathered[..., 3]
    
    """
    print("alpha max:", alpha.max().item(), "min:", alpha.min().item())
    print("phi max:", phi.max().item(), "min:", phi.min().item())
    print("omega_x max:", omega_x.max().item(), "min:", omega_x.min().item())
    print("omega_y max:", omega_y.max().item(), "min:", omega_y.min().item())
    """

    x_view = x_local.view(1, Hs, Ws, 1, 1)  # [1, Hs, Ws, 1, 1]
    y_view = y_local.view(1, Hs, Ws, 1, 1)

    # 4. 計算 V(x, y)
    value = alpha * torch.sin(omega_x * x_view + omega_y * y_view + phi)  # [B, Hs, Ws, C, n]
    images = value.mean(dim=-1).permute(0,3,1,2) + 0.5  # sum over n -> [B, C, Hs, Ws]

    return images
