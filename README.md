# VIT-on-imbalanced-CIFAR10

## 简介

由于单次上传文件数量有限，故本次实验只发送代码内容，完整项目浏览：https://github.com/mischievousx/VIT-on-imbalanced-CIFAR10

`net.py`: transformer模型

`preprocess.py`: 用于生成重采样数据

`pre_weight.py`: 用于类权重计算，在使用“train.py”的时候会自动调用

`test.py`: 测试模型，用于测试本次实验模型

`train.py`: 训练模型代码

`utils.py`: 包含数据、标签读取以及单个epoch的训练验证

`CIFAR10_imbalance`: 实验提供的训练、验证数据

`CIFAR10_imbalance_pre`: 重采样数据，该数据需要使用-“preprocess.py”生成

`CIFAR10_balance`: 实验提供的测试数据

`dataset.py`: 构建数据集 

## 类不均衡处理方式
- resample
- class_weighting

## 从头训练
- 确定"freeze_weight"为False.
- 本次实验使用的数据集为“CIFAR10”
- 模型参数修改如下：
```python
transformations = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)]),
    "val": transforms.Compose([
        transforms.Resize(36),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)])
}
```
```python
def create_model(num_classes: int = 10):
    return VisionTransformer(img_size=32, patch_size=4, embed_dim=48, depth=12, num_heads=12, num_classes=num_classes)
```
## 使用预训练模型

- 确定"freeze_weight"为True.
- 下载预训练模型'jx_vit_base_patch16_224_in21k-e5005f0a.pth'
- 模型参数修改如下：
```python
transformations = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)])
}
```
```python
def create_model(num_classes: int = 10):
    return VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=num_classes)
```
>ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
weights ported from official Google JAX impl:
https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
