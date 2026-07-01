from torch import nn
import torch
import numpy as np

class BBL(nn.Module):
    def __init__(
        self,
        Hl: int,
        Wl: int,
        init_sigma: float = 16.0,
    ):
        super().__init__()

        self.alpha = nn.Parameter(torch.full((1, 1, Hl, Wl), float(np.log(init_sigma)),   dtype=torch.float32))
        self.omega = nn.Parameter(torch.full((1, 1, Hl, Wl), float(np.log(init_sigma)),   dtype=torch.float32))
        self.th_raw = nn.Parameter(torch.zeros( (1, 1, Hl, Wl),                             dtype=torch.float32))
        
        
        