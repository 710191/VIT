import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


def UPA_SIN(HR_img, lr_modality, output_scale=1):
    """
    Main upsampling function using SIN parameters.
    Corresponds to UPA() in the original upsample_anything.py.
    
    Args:
        HR_img: PIL Image or numpy array
        lr_modality: torch.Tensor or numpy array (low-resolution features/image)
        output_scale: output size relative to HR_img. 1 -> same size as HR_img,
            2 -> output H/W doubled compared with HR_img.
    
    Returns:
        upsampled feature map [1, 3, Hh, Wh]
    """
    USE_AMP = True
    AMP_DTYPE = torch.float16
    
    # Prepare HR image
    hr = torch.from_numpy(np.array(HR_img)).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
    H, W = hr.shape[-2:]
    Hl, Wl = lr_modality.shape[-2:]
    target_h = int(H * output_scale)
    target_w = int(W * output_scale)
    scale = int(target_h / Hl)
    
    # Build the guide image at the requested output size
    guide_hr = F.interpolate(hr, size=(target_h, target_w), mode="bicubic", align_corners=False)
    
    # Build LR reference at the same spatial size as the input lr_modality
    lr = F.interpolate(hr, size=(Hl, Wl), mode="bicubic", align_corners=False)
    
    # Initialize model - corresponds to LearnablePixelwiseAnisoJBU_NoParent
    model = LearnablePixelwiseSIN_NoParent(Hl, Wl, scale=scale).cuda()
    model.train()
    
    # Optimizer and scheduler
    opt = torch.optim.Adam(model.parameters(), lr=1e-1)
    max_steps = 5100
    gamma = (1e-9 / 1e-1) ** (1.0 / max_steps)
    scheduler = LambdaLR(opt, lr_lambda=lambda step: gamma ** step)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    
    # Training loop
    for step in range(max_steps + 1):
        opt.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
            pred = model(lr, guide_hr)  # [1, 3, H_out, W_out]
            loss = F.l1_loss(pred, guide_hr)
        
        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        
        scheduler.step()
        
        if step == 50:
            break
    
    # Inference
    model.eval()
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=USE_AMP, dtype=AMP_DTYPE):
        hr_feat = model(lr_modality, guide_hr)
    
    return hr_feat


def gs_sin_upsample(
    feat_lr: torch.Tensor,      # [1, 3, Hl, Wl] LR feature map
    guide_hr: torch.Tensor,     # [1, 3, Hh, Wh] HR guide image
    alpha_map: torch.Tensor,    # [1, 1, Hl, Wl] amplitude
    omega_x_map: torch.Tensor,  # [1, 1, Hl, Wl] frequency in x
    omega_y_map: torch.Tensor,  # [1, 1, Hl, Wl] frequency in y
    phi_map: torch.Tensor,      # [1, 1, Hl, Wl] phase
    scale: int = 16,
    C_chunk: int = 512
):
    """
    SIN-based upsample core function: alpha * sin(omega_x * x + omega_y * y + phi)
    Corresponds to gs_jbu_aniso_noparent() in the original upsample_anything.py
    
    This is the rendering kernel that handles the actual upsampling computation.
    
    Args:
        feat_lr: [1, 3, Hl, Wl] low-resolution feature map
        guide_hr: [1, 3, Hh, Wh] high-resolution guide image
        alpha_map: [1, 1, Hl, Wl] amplitude parameter
        omega_x_map: [1, 1, Hl, Wl] x-frequency parameter
        omega_y_map: [1, 1, Hl, Wl] y-frequency parameter
        phi_map: [1, 1, Hl, Wl] phase parameter
        scale: upsampling scale factor
        C_chunk: channel chunk size for memory efficiency
    
    Returns:
        upsampled_feat: [1, 3, Hh, Wh]
    """
    
    _, C, Hl, Wl = feat_lr.shape
    _, _, Hh, Wh = guide_hr.shape
    device = feat_lr.device
    dtype = feat_lr.dtype
    
    # Create HR grid coordinates
    y_hr = torch.arange(Hh, device=device, dtype=torch.float32)
    x_hr = torch.arange(Wh, device=device, dtype=torch.float32)
    Y_hr, X_hr = torch.meshgrid(y_hr, x_hr, indexing='ij')  # [Hh, Wh]
    
    # Map HR coordinates to LR coordinates
    u = (Y_hr + 0.5) / scale - 0.5
    v = (X_hr + 0.5) / scale - 0.5
    
    # Clamp to valid LR range
    u_clamped = torch.clamp(u, 0, Hl - 1)
    v_clamped = torch.clamp(v, 0, Wl - 1)
    
    # Bilinear interpolation indices
    u_floor = torch.floor(u_clamped).long()
    u_ceil = (u_floor + 1).clamp(max=Hl - 1)
    v_floor = torch.floor(v_clamped).long()
    v_ceil = (v_floor + 1).clamp(max=Wl - 1)
    
    # Interpolation weights
    w_u = u_clamped - u_floor.float()
    w_v = v_clamped - v_floor.float()
    
    # Normalize coordinates to [0, 1] for SIN function
    x_norm = X_hr / Wh
    y_norm = Y_hr / Hh
    
    # Initialize output
    out = torch.zeros((1, C, Hh, Wh), device=device, dtype=dtype)
    
    # Process in chunks
    for c0 in range(0, C, C_chunk):
        c1 = min(c0 + C_chunk, C)
        
        # Get LR features for this chunk
        feat_chunk = feat_lr[:, c0:c1, :, :]  # [1, Cc, Hl, Wl]
        
        # Bilinear interpolate features
        feat_00 = feat_chunk[0, :, u_floor, v_floor]
        feat_01 = feat_chunk[0, :, u_floor, v_ceil]
        feat_10 = feat_chunk[0, :, u_ceil, v_floor]
        feat_11 = feat_chunk[0, :, u_ceil, v_ceil]
        
        feat_interp = (
            feat_00 * (1 - w_u) * (1 - w_v) +
            feat_01 * (1 - w_u) * w_v +
            feat_10 * w_u * (1 - w_v) +
            feat_11 * w_u * w_v
        )  # [Cc, Hh, Wh]
        
        # Get SIN parameters (shared across channels)
        alpha = torch.exp(alpha_map[0, 0, u_floor, v_floor])
        alpha_01 = torch.exp(alpha_map[0, 0, u_floor, v_ceil])
        alpha_10 = torch.exp(alpha_map[0, 0, u_ceil, v_floor])
        alpha_11 = torch.exp(alpha_map[0, 0, u_ceil, v_ceil])
        alpha = (
            alpha * (1 - w_u) * (1 - w_v) +
            alpha_01 * (1 - w_u) * w_v +
            alpha_10 * w_u * (1 - w_v) +
            alpha_11 * w_u * w_v
        )
        
        omega_x = omega_x_map[0, 0, u_floor, v_floor]
        omega_x_01 = omega_x_map[0, 0, u_floor, v_ceil]
        omega_x_10 = omega_x_map[0, 0, u_ceil, v_floor]
        omega_x_11 = omega_x_map[0, 0, u_ceil, v_ceil]
        omega_x = (
            omega_x * (1 - w_u) * (1 - w_v) +
            omega_x_01 * (1 - w_u) * w_v +
            omega_x_10 * w_u * (1 - w_v) +
            omega_x_11 * w_u * w_v
        )
        
        omega_y = omega_y_map[0, 0, u_floor, v_floor]
        omega_y_01 = omega_y_map[0, 0, u_floor, v_ceil]
        omega_y_10 = omega_y_map[0, 0, u_ceil, v_floor]
        omega_y_11 = omega_y_map[0, 0, u_ceil, v_ceil]
        omega_y = (
            omega_y * (1 - w_u) * (1 - w_v) +
            omega_y_01 * (1 - w_u) * w_v +
            omega_y_10 * w_u * (1 - w_v) +
            omega_y_11 * w_u * w_v
        )
        
        phi = phi_map[0, 0, u_floor, v_floor]
        phi_01 = phi_map[0, 0, u_floor, v_ceil]
        phi_10 = phi_map[0, 0, u_ceil, v_floor]
        phi_11 = phi_map[0, 0, u_ceil, v_ceil]
        phi = (
            phi * (1 - w_u) * (1 - w_v) +
            phi_01 * (1 - w_u) * w_v +
            phi_10 * w_u * (1 - w_v) +
            phi_11 * w_u * w_v
        )
        
        # Compute SIN: alpha * sin(omega_x * x + omega_y * y + phi)
        x_norm_unsq = x_norm.unsqueeze(0)
        y_norm_unsq = y_norm.unsqueeze(0)
        sin_value = alpha * torch.sin(omega_x * x_norm_unsq + omega_y * y_norm_unsq + phi)
        
        # Blend: interpolated feature * SIN modulation
        out[0, c0:c1, :, :] = feat_interp * sin_value
    
    return out


class LearnablePixelwiseSIN_NoParent(nn.Module):
    """
    Learnable pixel-wise SIN parameters for image upsampling.
    Corresponds to LearnablePixelwiseAnisoJBU_NoParent in the original upsample_anything.py
    
    Parameters:
        Hl, Wl: LR spatial dimensions
        scale: upsampling scale
    """
    
    def __init__(
        self,
        Hl: int,
        Wl: int,
        scale: int = 16,
        init_alpha: float = 0.0,
        init_omega: float = 2.0,
        init_phi: float = 0.0,
        scale_output: int = 1,
    ):
        super().__init__()
        self.Hl = Hl
        self.Wl = Wl
        self.scale = scale
        self.scale_output = scale_output
        
        # Alpha parameter (log-space for positivity)
        self.alpha_raw = nn.Parameter(
            torch.full((1, 1, Hl, Wl), float(init_alpha), dtype=torch.float32)
        )
        
        # Omega_x: frequency in x direction
        self.omega_x = nn.Parameter(
            torch.full((1, 1, Hl, Wl), float(init_omega), dtype=torch.float32)
        )
        
        # Omega_y: frequency in y direction
        self.omega_y = nn.Parameter(
            torch.full((1, 1, Hl, Wl), float(init_omega), dtype=torch.float32)
        )
        
        # Phi: phase shift
        self.phi = nn.Parameter(
            torch.full((1, 1, Hl, Wl), float(init_phi), dtype=torch.float32)
        )
    
    def forward(self, feat_lr: torch.Tensor, guide_hr: torch.Tensor):
        """
        Args:
            feat_lr: [1, 3, Hl, Wl] LR features (or downsampled)
            guide_hr: [1, 3, Hh, Wh] HR guide image
        
        Returns:
            upsampled: [1, 3, Hh, Wh]
        """
        Hh, Wh = guide_hr.shape[-2:]
        
        # Convert alpha from log space
        alpha = torch.exp(self.alpha_raw)
        
        return gs_sin_upsample(
            feat_lr=feat_lr,
            guide_hr=guide_hr,
            alpha_map=alpha,
            omega_x_map=self.omega_x,
            omega_y_map=self.omega_y,
            phi_map=self.phi,
            scale=int(Hh / self.Hl),
            C_chunk=512
        )


if __name__ == "__main__":
    print("UPA_SIN module loaded successfully")
