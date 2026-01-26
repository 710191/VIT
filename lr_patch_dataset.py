import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class LRPatchDataset(Dataset):
    def __init__(self, image_dir, image_list, crop_size=64, scale=1, num_iters=10, num_same_crop=20):
        self.image_dir = image_dir
        self.image_list = image_list
        self.crop_size = crop_size
        self.scale = scale
        self.num_iters = num_iters
        self.num_same_crop = num_same_crop
        self.to_tensor = transforms.ToTensor()

        # 預先生成所有 crop 的索引
        self.all_crops = []
        for img_name in self.image_list:
            lr_path = f"{self.image_dir}/{img_name}"
            hr_name = img_name.replace("_LR", "_HR")
            hr_path = f"{self.image_dir}/{hr_name}"

            lr_image = Image.open(lr_path).convert("RGB")
            hr_image = Image.open(hr_path).convert("RGB")
            lr_tensor = self.to_tensor(lr_image)
            hr_tensor = self.to_tensor(hr_image)

            _, H, W = lr_tensor.shape
            self.all_crops.extend(
                [
                    (img_name, x[0], x[1]) for x in zip(
                        torch.randint(0, H - self.crop_size + 1, (self.num_iters,)).tolist(),
                        torch.randint(0, W - self.crop_size + 1, (self.num_iters,)).tolist()
                    )
                ]
            )

    def __len__(self):
        return len(self.all_crops) * self.num_same_crop  # repeat num_same_crop times

    def __getitem__(self, idx):
        crop_idx = idx // self.num_same_crop
        img_name, top, left = self.all_crops[crop_idx]

        lr_path = f"{self.image_dir}/{img_name}"
        hr_name = img_name.replace("_LR", "_HR")
        hr_path = f"{self.image_dir}/{hr_name}"

        lr_image = Image.open(lr_path).convert("RGB")
        hr_image = Image.open(hr_path).convert("RGB")
        lr_tensor_full = self.to_tensor(lr_image)
        hr_tensor_full = self.to_tensor(hr_image)

        lr_crop = lr_tensor_full[:, top:top+self.crop_size, left:left+self.crop_size]
        hr_crop = hr_tensor_full[:, top*self.scale:(top+self.crop_size)*self.scale,
                                 left*self.scale:(left+self.crop_size)*self.scale]

        return lr_crop, hr_crop


class LRPatchDatasetPreloaded(Dataset):
    """
    Version of LRPatchDataset that preloads all images into RAM memory
    to avoid disk I/O on every __getitem__ call.
    """
    def __init__(self, image_dir, image_list, crop_size=64, scale=1, num_iters=100):
        self.image_dir = image_dir
        self.image_list = image_list
        self.crop_size = crop_size
        self.scale = scale
        self.num_iters = num_iters
        self.to_tensor = transforms.ToTensor()

        # Preload all images into memory
        self.preloaded_images = {}
        print(f"Preloading {len(image_list)} image pairs into RAM...")
        
        for img_name in self.image_list:
            lr_path = f"{self.image_dir}/{img_name}"
            hr_name = img_name.replace("_LR", "_HR")
            hr_path = f"{self.image_dir}/{hr_name}"

            lr_image = Image.open(lr_path).convert("RGB")
            hr_image = Image.open(hr_path).convert("RGB")
            lr_tensor = self.to_tensor(lr_image)
            hr_tensor = self.to_tensor(hr_image)

            # Store tensors in memory
            self.preloaded_images[img_name] = {
                'lr': lr_tensor,
                'hr': hr_tensor
            }
        
        print(f"Preloading complete. {len(self.preloaded_images)} image pairs loaded.")

    def __len__(self):
        return len(self.image_list) * self.num_iters

    def __getitem__(self, idx):
        # Randomly select an image
        img_idx = torch.randint(0, len(self.image_list), (1,)).item()
        img_name = self.image_list[img_idx]

        # Get preloaded tensors from memory (no disk I/O)
        lr_tensor_full = self.preloaded_images[img_name]['lr']
        hr_tensor_full = self.preloaded_images[img_name]['hr']

        # Randomly generate crop position
        _, H, W = lr_tensor_full.shape
        top = torch.randint(0, H - self.crop_size + 1, (1,)).item()
        left = torch.randint(0, W - self.crop_size + 1, (1,)).item()

        lr_crop = lr_tensor_full[:, top:top+self.crop_size, left:left+self.crop_size]
        hr_crop = hr_tensor_full[:, top*self.scale:(top+self.crop_size)*self.scale,
                                 left*self.scale:(left+self.crop_size)*self.scale]

        return lr_crop, hr_crop