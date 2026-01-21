import torch
import torch.nn as nn
import torch.nn.functional as F
from SIREN import SineLayer

class SIREN_MLP(nn.Module):
    """
    通用 MLP: input_dim -> output_dim
    可以用於 patch token 或其他 latent
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 1024, 512]):
        super().__init__()
        self.fc1 = SineLayer(input_dim, hidden_dims[0], is_first=True, omega_0=10)
        self.fc2 = SineLayer(hidden_dims[0], hidden_dims[1], is_first=False, omega_0=10)
        self.fc3 = SineLayer(hidden_dims[1], hidden_dims[2], is_first=False, omega_0=10)
        self.fc_out = nn.Linear(hidden_dims[2], output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        out = self.fc_out(x)
        return out