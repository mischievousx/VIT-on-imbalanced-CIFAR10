# VIT-on-imbalanced-CIFAR10

## Introduction

`net.py`: VIT model

`preprocess.py`: Used to generate resampling data

`pre_weight.py`: For class weighting, automatically called when using “train.py”

`test.py`: Test model for testing this experimental model

`train.py`: Training model code

`utils.py`: Includes data, labeled reads, and training validation for individual epochs

`CIFAR10_imbalance`: Experiments provide training, validation data

`CIFAR10_imbalance_pre`: Resampling data, which needs to be generated using - “preprocess.py”

`CIFAR10_balance`: Experiments provide test data

`dataset.py`: Building data sets

## Class Imbalance Handling
- resample
- class_weighting

## Scratch Training
- Set “freeze_weight” to False.
- The dataset used in this experiment is “CIFAR10”.
- The model parameters are modified as follows:
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
## Using pre-trained models

- Make sure 'freeze_weight' is True.
- Download the pre-trained model 'jx_vit_base_patch16_224_in21k-e5005f0a.pth'.
- Modify the model parameters as follows:
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
