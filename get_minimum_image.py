import os
from PIL import Image

def find_min_hw(folder_path):
    min_h, min_w = float('inf'), float('inf')

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".png"):
            path = os.path.join(folder_path, filename)
            with Image.open(path) as img:
                w, h = img.size  # PIL: size = (width, height)
                min_h = min(min_h, h)
                min_w = min(min_w, w)

    if min_h == float('inf') or min_w == float('inf'):
        raise ValueError("No PNG files found in folder.")

    return min_h, min_w

# 範例使用
folder_path = "../dataset/DrealSR"
min_height, min_width = find_min_hw(folder_path)
print("最小高度 H:", min_height)
print("最小寬度 W:", min_width)
