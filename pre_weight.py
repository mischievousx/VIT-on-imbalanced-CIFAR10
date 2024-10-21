import os
import torch

def calculate_weight(device):
    # CIFAR-10 文件夹路径
    data_dir = r"./CIFAR10_imbalance"
    num_images = []

    # 遍历 0 到 9 的文件夹，并统计每个文件夹中的图片数量
    for i in range(10):
        folder_path = os.path.join(data_dir, str(i))
        if os.path.exists(folder_path):
            count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            num_images.append(count)  # 记录每个类的样本数量

    # 计算总样本数
    total_samples = sum(num_images)

    # 计算类权重
    class_weights = [total_samples / (len(num_images) * count) for count in num_images]

    # 将类权重转换为 torch Tensor
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    class_weights_tensor = class_weights_tensor.to(device)

    return class_weights_tensor

