import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):
    """
    SIREN backbone + multi-head for fft_parameters
    Each head predicts one fft parameter independently
    """
    def __init__(self, input_dim, colors, n, fft_parameters, hidden_dims=[1024, 1024, 1024]):
        super().__init__()
        self.colors = colors
        self.n = n
        self.fft_parameters = fft_parameters

        # Shared SIREN backbone
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])

        # Multi-head: one head per fft parameter
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dims[2], colors * n) for _ in range(fft_parameters)
        ])
        self.amplitude_IDS = {0}   # amplitude heads
        self.phase_IDS = {1}   # phase heads
        self.omega_IDS = {2} # omega_x, omega_y heads

    def forward(self, x):
        # Shared backbone
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Forward through each head independently
        outputs = []
        for i, head in enumerate(self.heads):
            out = head(x)  # [B, colors * n]

            # ===== head-wise activation =====
            if i in self.amplitude_IDS:
                out = torch.sigmoid(out)                # (0, 1)
            elif i in self.phase_IDS:
                out = torch.tanh(out) * math.pi         # (−π, π)
            elif i in self.omega_IDS:
                out = F.softplus(out)                   # (0, inf)
            # else: 不限制（coef / latent / residual）

            outputs.append(out)

        # Concatenate along the fft_parameter dimension
        # Final shape: [batch, colors * n * fft_parameters]
        out = torch.cat(outputs, dim=-1)
        return out