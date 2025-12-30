import os
from PIL import Image

def preprocess_images_by_suffix(folder_path, out_folder, patch_size=32, scale=2):
    """
    根據檔名後綴 _LR / _HR 判斷
    將所有 PNG resize 成 patch_size 的倍數
    LR resize → [H, W]
    HR resize → [H*scale, W*scale]
    存到 out_folder，保留原始檔名
    """
    os.makedirs(out_folder, exist_ok=True)

    # 找最小 H, W 只考慮 LR 圖片
    min_h, min_w = float('inf'), float('inf')
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".png") and "_LR" in filename:
            path = os.path.join(folder_path, filename)
            with Image.open(path) as img:
                w, h = img.size
                min_h = min(min_h, h)
                min_w = min(min_w, w)

    # 調整成 patch_size 的倍數
    min_h = (min_h // patch_size) * patch_size
    min_w = (min_w // patch_size) * patch_size
    print(f"統一大小 LR: {min_h}x{min_w}, HR: {min_h*scale}x{min_w*scale}")

    # 遍歷資料夾 resize 並存檔
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".png"):
            path = os.path.join(folder_path, filename)
            with Image.open(path) as img:
                if "_LR" in filename:
                    lr_img = img.resize((min_w, min_h), Image.BICUBIC)
                    lr_img.save(os.path.join(out_folder, filename))
                elif "_HR" in filename:
                    hr_img = img.resize((min_w*scale, min_h*scale), Image.BICUBIC)
                    hr_img.save(os.path.join(out_folder, filename))


preprocess_images_by_suffix(
    folder_path="../dataset/DrealSR",
    out_folder="../dataset/DrealSR_cut",
    patch_size=32,
    scale=2
)