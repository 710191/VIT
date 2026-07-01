import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


def upsample_sin(
    feat_lr: torch.Tensor,      # [1, C, Hl, Wl] LR feature map
    guide_hr: torch.Tensor,     # [1, 3, Hh, Wh] HR guide image
    alpha_map: torch.Tensor,    # [1, C, Hl, Wl] amplitude
    omega_x_map: torch.Tensor,  # [1, C, Hl, Wl] frequency in x
    omega_y_map: torch.Tensor,  # [1, C, Hl, Wl] frequency in y
    phi_map: torch.Tensor,      # [1, C, Hl, Wl] phase
    scale: int = 16,
    C_chunk: int = 128,
):
    """
    SIN-based upsample: alpha * sin(omega_x * x + omega_y * y + phi)
    
    Args:
        feat_lr: [1, C, Hl, Wl]
        guide_hr: [1, 3, Hh, Wh]
        alpha_map: [1, C, Hl, Wl]
        omega_x_map: [1, C, Hl, Wl]
        omega_y_map: [1, C, Hl, Wl]
        phi_map: [1, C, Hl, Wl]
        scale: upsampling scale factor
        C_chunk: channel chunk size for memory efficiency
    
    Returns:
        upsampled_feat: [1, C, Hh, Wh]
    """
    
    _, C, Hl, Wl = feat_lr.shape
    _, _, Hh, Wh = guide_hr.shape
    device = feat_lr.device
    dtype = feat_lr.dtype
    
    # Create HR grid coordinates
    y_hr = torch.arange(Hh, device=device, dtype=torch.float32)
    x_hr = torch.arange(Wh, device=device, dtype=torch.float32)
    Y_hr, X_hr = torch.meshgrid(y_hr, x_hr, indexing='ij')  # [Hh, Wh]
    
    # Map HR coordinates to LR coordinates (normalized)
    u = (Y_hr + 0.5) / scale - 0.5  # [Hh, Wh]
    v = (X_hr + 0.5) / scale - 0.5  # [Hh, Wh]
    
    # Clamp to valid LR range [0, Hl-1] and [0, Wl-1]
    u_clamped = torch.clamp(u, 0, Hl - 1)  # [Hh, Wh]
    v_clamped = torch.clamp(v, 0, Wl - 1)  # [Hh, Wh]
    
    # Get floor and ceil indices for bilinear interpolation
    u_floor = torch.floor(u_clamped).long()
    u_ceil = (u_floor + 1).clamp(max=Hl - 1)
    v_floor = torch.floor(v_clamped).long()
    v_ceil = (v_floor + 1).clamp(max=Wl - 1)
    
    # Interpolation weights
    w_u = u_clamped - u_floor.float()  # [Hh, Wh]
    w_v = v_clamped - v_floor.float()  # [Hh, Wh]
    
    # Normalize coordinates to [0, 1] for SIN function
    x_norm = (X_hr) / Wh  # [Hh, Wh]
    y_norm = (Y_hr) / Hh  # [Hh, Wh]
    
    # Initialize output
    out = torch.zeros((1, C, Hh, Wh), device=device, dtype=dtype)
    
    # Process in chunks to save memory
    for c0 in range(0, C, C_chunk):
        c1 = min(c0 + C_chunk, C)
        c_len = c1 - c0
        
        # Get parameters for this chunk
        alpha = alpha_map[:, c0:c1, :, :]  # [1, Cc, Hl, Wl]
        omega_x = omega_x_map[:, c0:c1, :, :]  # [1, Cc, Hl, Wl]
        omega_y = omega_y_map[:, c0:c1, :, :]  # [1, Cc, Hl, Wl]
        phi = phi_map[:, c0:c1, :, :]  # [1, Cc, Hl, Wl]
        
        # Bilinear interpolation of parameters
        alpha_00 = alpha[0, :, u_floor, v_floor]  # [Cc, Hh, Wh]
        alpha_01 = alpha[0, :, u_floor, v_ceil]
        alpha_10 = alpha[0, :, u_ceil, v_floor]
        alpha_11 = alpha[0, :, u_ceil, v_ceil]
        
        alpha_interp = (
            alpha_00 * (1 - w_u) * (1 - w_v) +
            alpha_01 * (1 - w_u) * w_v +
            alpha_10 * w_u * (1 - w_v) +
            alpha_11 * w_u * w_v
        )  # [Cc, Hh, Wh]
        
        omega_x_00 = omega_x[0, :, u_floor, v_floor]
        omega_x_01 = omega_x[0, :, u_floor, v_ceil]
        omega_x_10 = omega_x[0, :, u_ceil, v_floor]
        omega_x_11 = omega_x[0, :, u_ceil, v_ceil]
        
        omega_x_interp = (
            omega_x_00 * (1 - w_u) * (1 - w_v) +
            omega_x_01 * (1 - w_u) * w_v +
            omega_x_10 * w_u * (1 - w_v) +
            omega_x_11 * w_u * w_v
        )  # [Cc, Hh, Wh]
        
        omega_y_00 = omega_y[0, :, u_floor, v_floor]
        omega_y_01 = omega_y[0, :, u_floor, v_ceil]
        omega_y_10 = omega_y[0, :, u_ceil, v_floor]
        omega_y_11 = omega_y[0, :, u_ceil, v_ceil]
        
        omega_y_interp = (
            omega_y_00 * (1 - w_u) * (1 - w_v) +
            omega_y_01 * (1 - w_u) * w_v +
            omega_y_10 * w_u * (1 - w_v) +
            omega_y_11 * w_u * w_v
        )  # [Cc, Hh, Wh]
        
        phi_00 = phi[0, :, u_floor, v_floor]
        phi_01 = phi[0, :, u_floor, v_ceil]
        phi_10 = phi[0, :, u_ceil, v_floor]
        phi_11 = phi[0, :, u_ceil, v_ceil]
        
        phi_interp = (
            phi_00 * (1 - w_u) * (1 - w_v) +
            phi_01 * (1 - w_u) * w_v +
            phi_10 * w_u * (1 - w_v) +
            phi_11 * w_u * w_v
        )  # [Cc, Hh, Wh]
        
        # Compute SIN value: alpha * sin(omega_x * x + omega_y * y + phi)
        x_norm_expanded = x_norm.unsqueeze(0)  # [1, Hh, Wh]
        y_norm_expanded = y_norm.unsqueeze(0)  # [1, Hh, Wh]
        
        sin_value = alpha_interp * torch.sin(
            omega_x_interp * x_norm_expanded + 
            omega_y_interp * y_norm_expanded + 
            phi_interp
        )  # [Cc, Hh, Wh]
        
        # Clamp to [-0.5, 0.5] and shift to [0, 1]
        sin_value = torch.clamp(sin_value, -0.5, 0.5)
        
        # Put into output
        out[0, c0:c1, :, :] = sin_value
    
    return out


class LearnablePixelwiseSIN(nn.Module):
    """
    Learnable pixel-wise SIN parameters for image upsampling
    
    Parameters:
        Hl, Wl: LR spatial dimensions
        C: number of channels
        scale: upsampling scale
    """
    
    def __init__(
        self,
        Hl: int,
        Wl: int,
        C: int = 64,
        scale: int = 16,
        init_alpha: float = 0.5,
        init_omega: float = 2.0,
    ):
        super().__init__()
        self.Hl = Hl
        self.Wl = Wl
        self.C = C
        self.scale = scale
        
        # Initialize parameters
        # alpha: amplitude
        self.alpha = nn.Parameter(
            torch.full((1, C, Hl, Wl), float(init_alpha), dtype=torch.float32)
        )
        
        # omega_x: frequency in x direction
        self.omega_x = nn.Parameter(
            torch.full((1, C, Hl, Wl), float(init_omega), dtype=torch.float32)
        )
        
        # omega_y: frequency in y direction
        self.omega_y = nn.Parameter(
            torch.full((1, C, Hl, Wl), float(init_omega), dtype=torch.float32)
        )
        
        # phi: phase shift
        self.phi = nn.Parameter(
            torch.zeros((1, C, Hl, Wl), dtype=torch.float32)
        )
    
    def forward(self, feat_lr: torch.Tensor, guide_hr: torch.Tensor):
        """
        Args:
            feat_lr: [1, C, Hl, Wl] LR features
            guide_hr: [1, 3, Hh, Wh] HR guide image
        
        Returns:
            upsampled: [1, C, Hh, Wh]
        """
        Hh, Wh = guide_hr.shape[-2:]
        
        return upsample_sin(
            feat_lr=feat_lr,
            guide_hr=guide_hr,
            alpha_map=self.alpha,
            omega_x_map=self.omega_x,
            omega_y_map=self.omega_y,
            phi_map=self.phi,
            scale=int(Hh / self.Hl),
            C_chunk=self.C,
        )


def UPA_SIN(HR_img, lr_modality, num_channels=64):
    """
    Main upsampling function using SIN parameters
    
    Args:
        HR_img: PIL Image or numpy array
        lr_modality: torch.Tensor or numpy array
        num_channels: number of feature channels
    
    Returns:
        upsampled feature map [1, num_channels, Hh, Wh]
    """
    USE_AMP = True
    AMP_DTYPE = torch.float16
    
    # Prepare HR image
    if isinstance(HR_img, np.ndarray):
        hr = torch.from_numpy(HR_img).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
    else:
        hr = torch.from_numpy(np.array(HR_img)).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
    
    H, W = hr.shape[-2:]
    Hl, Wl = lr_modality.shape[-2:]
    scale = int(H / Hl)
    
    # Downsample HR as LR reference
    lr = F.interpolate(hr, scale_factor=1/scale, mode="bicubic", align_corners=False)
    
    # Initialize model
    model = LearnablePixelwiseSIN(Hl, Wl, C=num_channels, scale=scale).cuda()
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
            # For training, use dummy LR features
            lr_feat = F.interpolate(lr, size=(Hl, Wl), mode="bilinear", align_corners=False)
            pred = model(lr_feat, hr)  # [1, num_channels, H, W]
            
            # Simple loss: match to HR guide
            if pred.shape[1] == 3:
                loss = F.l1_loss(pred, hr)
            else:
                # For multi-channel, average the loss
                loss = F.l1_loss(pred[:, :3], hr)
        
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
        if isinstance(lr_modality, np.ndarray):
            lr_feat_input = torch.from_numpy(lr_modality).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
        else:
            lr_feat_input = lr_modality
        
        # Prepare LR features
        if lr_feat_input.ndim == 4 and lr_feat_input.shape[1] == 3:
            # Convert RGB to feature space (simple interpolation for now)
            lr_feat_input = F.interpolate(
                lr_feat_input, 
                size=(Hl, Wl), 
                mode="bilinear", 
                align_corners=False
            )
        
        hr_feat = model(lr_feat_input, hr)
    
    return hr_feat


if __name__ == "__main__":
    # Simple test
    print("UPA_SIN module loaded successfully")
