import torch
print(torch.__version__)          # >= 2.6
print(torch.version.cuda)         # CUDA 12.4
print(torch.cuda.is_available())  # True