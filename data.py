import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

class OverfitDataset(Dataset):
    def __init__(self, image_paths, size=None):
        """
        Args:
            image_paths (str): Path to the image file.
        """
        self.image = Image.open(image_paths).convert('RGB')
        if size is not None:
            self.image = self.image.resize((size, size), Image.LANCZOS)
        self.image = torch.from_numpy(np.array(self.image)).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1

    def __len__(self):
        return 1  # Only one image

    def __getitem__(self, idx):
        return self.image

class GeneralDataset(Dataset):
    def __init__(self, image_paths):
        """
        Args:
            image_paths (list of str): A list of paths to image files.
        """
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]
        return image
