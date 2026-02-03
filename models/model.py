import torch
import torch.nn as nn

from .hat import HAT

class FFTPredictor(nn.Module):
    """
    Implicit Neural Representation model that predicts FFT coefficients from RGB input.
    Uses HAT as backbone which outputs features of shape [B, 64, H*2, W*2].
    """
    def __init__(
        self, 
        checkpoint,
        hidden_dim=256, 
        output_channels=6,
    ):
        super().__init__()
        
        # HAT backbone: [B, 3, H, W] -> [B, 64, H*2, W*2]
        self.backbone = HAT(
            upscale=2,
            in_chans=3,
            img_size=64,
            window_size=16,
            compress_ratio=3,
            squeeze_factor=30,
            conv_scale=0.01,
            overlap_ratio=0.5,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv',
        )
        self.backbone.load_state_dict(
            torch.load(checkpoint)['params_ema'],
        )

        # Projection head: [B, 64, H*2, W*2] -> [B, 6, H, W]
        # First reduce channels, then downsample spatially
        self.head = nn.Sequential(
            nn.PixelUnshuffle(2),  # [B, 64, H*2, W*2] -> [B, 256, H, W]
            nn.Conv2d(256, hidden_dim, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, output_channels, kernel_size=1, padding=0),
        )

    def forward(self, x):
        """
        Args:
            x: RGB image tensor [B, 3, H, W]
        
        Returns:
            fft_pred: Predicted FFT coefficients [B, 6, H, W]
                      First 3 channels: real parts
                      Last 3 channels: imaginary parts
        """
        # Extract features using HAT backbone: [B, 3, H, W] -> [B, 64, H*2, W*2]
        features = self.backbone(x)
        # Project to FFT coefficients: [B, 64, H*2, W*2] -> [B, 6, H, W]
        fft_pred = self.head(features)
        
        return fft_pred
    
    def predict_rgb(self, fft_pred):
        """
        Reconstruct RGB image from predicted FFT coefficients.
        
        Args:
            fft_pred: Predicted FFT coefficients [B, 6, H, W]
        
        Returns:
            rgb_recon: Reconstructed RGB image [B, 3, H, W]
        """
        B, C, H, W = fft_pred.shape
        
        # Split real and imaginary parts
        real = fft_pred[:, :3, :, :]  # [B, 3, H, W]
        imag = fft_pred[:, 3:, :, :]  # [B, 3, H, W]
        
        # Combine into complex tensor
        fft_complex = torch.complex(real, imag)  # [B, 3, H, W]
        
        # Inverse FFT to get RGB image
        rgb_recon = torch.fft.ifft2(fft_complex, norm='ortho').real
        
        return rgb_recon

def compute_fft(rgb_image):
    """
    Compute FFT coefficients from RGB image.
    
    Args:
        rgb_image: RGB image tensor [B, 3, H, W]
    
    Returns:
        fft_coeffs: FFT coefficients [B, 6, H, W]
                    First 3 channels: real parts
                    Last 3 channels: imaginary parts
    """
    # Compute 2D FFT
    fft_complex = torch.fft.fft2(rgb_image, norm='ortho')  # [B, 3, H, W]
    
    # Split into real and imaginary parts
    real = fft_complex.real
    imag = fft_complex.imag
    
    # Concatenate along channel dimension
    fft_coeffs = torch.cat([real, imag], dim=1)  # [B, 6, H, W]
    
    return fft_coeffs
