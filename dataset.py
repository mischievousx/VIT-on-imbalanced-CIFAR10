from PIL import Image
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    # 初始化自定义数据集
    def __init__(self, image_paths: list, image_labels: list, transform=None):
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.transform = transform

    # 返回数据集中样本的数量
    def __len__(self):
        return len(self.image_paths)

    # 根据索引获取图像和标签
    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        if img.mode != 'RGB':
            raise ValueError(f"Image at {self.image_paths[index]} is not in RGB mode.")
        label = self.image_labels[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    # 自定义合并函数，用于将一个批次的样本合并为一个 tensor
    @staticmethod
    def custom_collate_fn(batch):
        imgs, lbls = tuple(zip(*batch))
        imgs = torch.stack(imgs, dim=0)
        lbls = torch.tensor(lbls)
        return imgs, lbls
