import os
import sys
import json
import pickle
import random
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from pre_weight import calculate_weight

def load_train_data(roots: list, val_rate: float = 0.2):
    random.seed(0)  # 保持随机选择的可复现性
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    class_counts, supported_formats = [], [".jpg", ".JPG", ".png", ".PNG"]
    
    # 提取所有类别
    all_classes = {folder for root in roots if os.path.exists(root) for folder in os.listdir(root) if os.path.isdir(os.path.join(root, folder))}
    class_list = sorted(all_classes)
    class_indices = {name: idx for idx, name in enumerate(class_list)}

    # 保存类别索引至 JSON 文件
    with open('class_indices.json', 'w') as file:
        json.dump({v: k for k, v in class_indices.items()}, file, indent=4)

    # 遍历每个目录，分配到训练集或验证集
    for root in roots:
        for cls in class_list:
            cls_path = os.path.join(root, cls)
            if not os.path.exists(cls_path):
                continue
            images = [os.path.join(cls_path, img) for img in os.listdir(cls_path) if os.path.splitext(img)[-1] in supported_formats]
            images.sort()
            class_idx = class_indices[cls]
            class_counts.append(len(images))
            val_samples = random.sample(images, k=int(len(images) * val_rate))

            # 分配数据到训练集和验证集
            for img_path in images:
                if img_path in val_samples:
                    val_paths.append(img_path)
                    val_labels.append(class_idx)
                else:
                    train_paths.append(img_path)
                    train_labels.append(class_idx)

    print(f"Found {sum(class_counts)} images.")
    print(f"Training set size: {len(train_paths)}, Validation set size: {len(val_paths)}")
    assert train_paths, "Training set cannot be empty."
    assert val_paths, "Validation set cannot be empty."
    return train_paths, train_labels, val_paths, val_labels

def load_test_data(root: str):
    assert os.path.exists(root), f"Dataset root '{root}' does not exist."
    class_list = sorted([folder for folder in os.listdir(root) if os.path.isdir(os.path.join(root, folder))])
    class_indices = {name: idx for idx, name in enumerate(class_list)}
    # 保存类别索引至 JSON 文件
    with open('class_indices.json', 'w') as file:
        json.dump({v: k for k, v in class_indices.items()}, file, indent=4)
    test_paths, test_labels, class_counts, supported_formats = [], [], [], [".jpg", ".JPG", ".png", ".PNG"]

    # 遍历文件
    for cls in class_list:
        cls_path = os.path.join(root, cls)
        images = [os.path.join(cls_path, img) for img in os.listdir(cls_path) if os.path.splitext(img)[-1] in supported_formats]
        images.sort()
        class_idx = class_indices[cls]
        class_counts.append(len(images))
        
        for img_path in images:
            test_paths.append(img_path)
            test_labels.append(class_idx)

    print(f"Found {sum(class_counts)} test images.")
    assert test_paths, "Test set cannot be empty."
    return test_paths, test_labels

def train_one_epoch(model, optimizer, data_loader, device, epoch, use_weights=False):
    weights = calculate_weight(device) if use_weights else None
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    total_loss, correct_preds, total_samples = 0.0, 0, 0
    optimizer.zero_grad()
    data_loader = tqdm(data_loader, file=sys.stdout)
    
    for step, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        predictions = model(images)
        loss = loss_fn(predictions, labels)
        loss.backward()
        if not torch.isfinite(loss):
            print("Non-finite loss encountered. Stopping training.")
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        correct_preds += (predictions.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)
        data_loader.desc = f"[Epoch {epoch}] Loss: {total_loss / (step + 1):.3f}, Accuracy: {correct_preds / total_samples:.3f}"
    
    return total_loss / (step + 1), correct_preds / total_samples

@torch.no_grad()
def val_one_epoch(model, data_loader, device, epoch, use_weights=False):
    weights = calculate_weight(device) if use_weights else None
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    total_loss, correct_preds, total_samples = 0.0, 0, 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    
    for step, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        predictions = model(images)
        loss = loss_fn(predictions, labels)
        
        total_loss += loss.item()
        correct_preds += (predictions.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)
        
        data_loader.desc = f"[Validation Epoch {epoch}] Loss: {total_loss / (step + 1):.3f}, Accuracy: {correct_preds / total_samples:.3f}"
    
    return total_loss / (step + 1), correct_preds / total_samples