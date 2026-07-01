import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from lr_patch_dataset import LRPatchDataset, LRPatchDatasetPreloaded
from tqdm import tqdm
from src.upsample_anything_SIN import UPA_SIN
from torchvision.utils import save_image
import torchvision.transforms as T
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UPA_SIN

# PSNR 計算函數
def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# training: 02~83
image_dir = "/work/a28133781/datasets/DrealSR_cut"
image_list = [f"DrealSR{str(i).zfill(2)}_LR.png" for i in range(1, 2)]

to_tensor = T.ToTensor()

for image_name in tqdm(image_list):
    lr_path = os.path.join(image_dir, image_name)
    hr_path = os.path.join(image_dir, image_name.replace("_LR.png", "_HR.png"))
    
    lr_img = Image.open(lr_path).convert("RGB")
    hr_img = Image.open(hr_path).convert("RGB")
    
    lr_tensor = to_tensor(lr_img).to(device).unsqueeze(0)
    hr_tensor = to_tensor(hr_img).to(device).unsqueeze(0)
    
    
    # to prevent too eazy
    print(f"Original LR size: {lr_tensor.shape}, Original HR size: {hr_tensor.shape}")
    lr_img = lr_img.resize((256, 256), Image.BICUBIC)
    lr_tensor = F.interpolate(lr_tensor, size=(256, 256), mode='bicubic', align_corners=False)
    hr_tensor = F.interpolate(hr_tensor, size=(1024, 1024), mode='bicubic', align_corners=False)
    print(f"Resized LR size: {lr_tensor.shape}, Resized HR size: {hr_tensor.shape}")
    image_name = image_name.replace("_LR.png", "_resized_LR.png")
    
    
    lr_predict = model(lr_img, lr_tensor)
    lr_epoch_psnr = psnr(lr_predict, lr_tensor)
    hr_predict = F.interpolate(lr_predict, scale_factor=2, mode="bicubic", align_corners=False)
    hr_epoch_psnr = psnr(hr_predict, hr_tensor)
    print(f"LR PSNR: {lr_epoch_psnr:.4f} dB, HR PSNR: {hr_epoch_psnr:.4f} dB") #44 33

    save_image(lr_tensor, f"./output/SIN/lr_{image_name}.png")
    save_image(hr_tensor, f"./output/SIN/hr_{image_name}.png")
    save_image(lr_predict, f"./output/SIN/UPA_{image_name}.png")
    save_image(hr_predict, f"./output/SIN/UPA_hr_{image_name}.png")