import os
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
import random

# 路径定义
root_dir = './CIFAR10_imbalance'
output_dir = './CIFAR10_imbalance_pre'

# 数据增强的变换操作
transform_augment = transforms.Compose([
    transforms.RandomRotation(30),   # 随机旋转
    transforms.ColorJitter(contrast=0.5),  # 对比度增强
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),  # 随机裁剪和缩放
])

# 遍历每个类别的文件夹
for label in range(10):
    folder_path = os.path.join(root_dir, str(label))
    output_path = os.path.join(output_dir, str(label))
    os.makedirs(output_path, exist_ok=True)

    image_files = os.listdir(folder_path)
    num_images = len(image_files)
    max_images = 4513  # 定义目标最大图片数

    if num_images < max_images:
        # 增加样本数到 max_images
        for i in range(max_images - num_images):
            img_name = random.choice(image_files)
            img_path = os.path.join(folder_path, img_name)

            # 加载图像并应用数据增强
            image = datasets.folder.default_loader(img_path)
            augmented_image = transform_augment(image)

            # 保存增强后的图像
            save_image(transforms.ToTensor()(augmented_image), os.path.join(output_path, f'aug_{i}.png'))

print("Data augmentation completed!")