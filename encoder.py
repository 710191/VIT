import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

from hat import HAT

class Encoder(nn.Module):
    def __init__(self, name, path):
        super(Encoder, self).__init__()

        args = {'upscale': 2,
            'in_chans': 3,
            'img_size': 64,
            'window_size': 16,
            'compress_ratio': 3,
            'squeeze_factor': 30,
            'conv_scale': 0.01,
            'overlap_ratio': 0.5,
            'img_range': 1.,
            'depths': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            'embed_dim': 180,
            'num_heads': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            'mlp_ratio': 2,
            'upsampler': 'pixelshuffle',
            'resi_connection': '1conv'
        }

        if name == 'HAT':
            self.model = HAT(**args)
            if path:
                state_dict = torch.load(path)
                self.model.load_state_dict(state_dict['params_ema'], strict=True)
        else:
            raise NotImplementedError(f'Encoder {name} is not implemented.')

    def forward(self, x):
        return self.model(x)

    def forward_features(self, x):
        return self.model.forward_features(x)

    def conv_first(self, x):
        return self.model.conv_first(x)