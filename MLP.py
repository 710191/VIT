import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    通用 MLP: input_dim -> output_dim
    可以用於 patch token 或其他 latent
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 1024, 512]):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
         
        # Shared SIREN backbone
        self.fc1 = SineLayer(input_dim, hidden_dims[0], is_first=True, omega_0=10)
        self.fc2 = SineLayer(hidden_dims[0], hidden_dims[1], is_first=False, omega_0=10)
        self.fc3 = SineLayer(hidden_dims[1], hidden_dims[2], is_first=False, omega_0=10)

        # Multi-head: one head per fft parameter
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dims[2], colors * n) for _ in range(fft_parameters)
        ])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # Forward through each head independently
        outputs = []
        for head in self.heads:
            out = head(x)  # [batch, colors * n]
            outputs.append(out)

        # Concatenate along the fft_parameter dimension
        # Final shape: [batch, colors * n * fft_parameters]
        out = torch.cat(outputs, dim=-1)
        return out