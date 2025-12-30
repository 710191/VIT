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
        self.fc_out = nn.Linear(hidden_dims[2], output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = self.fc_out(x)
        return out