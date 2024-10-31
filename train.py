import os
import math
import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from dataset import MyDataset
from net import create_model
from utils import load_train_data, train_one_epoch, val_one_epoch

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.9)
    parser.add_argument('--orign_data_path', type=str, default="./CIFAR10_imbalance")
    parser.add_argument('--resample_data_path', type=str, default="./CIFAR10_imbalance_pre")
    parser.add_argument('--model_name', default='', help='create model name')
    parser.add_argument('--preprocess', type=str, default="resample")
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--freeze_layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    return parser.parse_args()

def run_training(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if not os.path.exists("./model_weights"):
        os.makedirs("./model_weights")
    tensorboard_writer = SummaryWriter()

    # 类不均衡问题处理
    data_sources = [args.orign_data_path]
    if args.preprocess == "resample":
        data_sources.append(args.resample_data_path)
    use_class_weighting = args.preprocess == "class_weighting"

    # 加载获取训练、验证数据以及标签
    train_paths, train_labels, val_paths, val_labels = load_train_data(data_sources)

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

    # 初始化数据集
    training_dataset = MyDataset(image_paths=train_paths, image_labels=train_labels, transform=transformations["train"])
    validation_dataset = MyDataset(image_paths=val_paths, image_labels=val_labels, transform=transformations["val"])

    batch_size = args.batch_size
    
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {num_workers} dataloader workers for each process.')

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, collate_fn=training_dataset.custom_collate_fn)
    val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers, collate_fn=validation_dataset.custom_collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)

    if args.weights:
        assert os.path.exists(args.weights), f"weights file: '{args.weights}' does not exist."
        weights_dict = torch.load(args.weights, map_location=device)
        # 将不必要的键值从权重当中移除
        for key in ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']:
            weights_dict.pop(key, None)
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, param in model.named_parameters():
            param.requires_grad = "head" in name or "pre_logits" in name
            if param.requires_grad:
                print(f"Training layer: {name}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=5E-5)
    scheduler_lambda = lambda epoch: ((1 + math.cos(epoch * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # Cosine learning rate scheduler
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_lambda)

    for epoch in range(args.epochs):
        # 训练Train
        train_loss, train_accuracy = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader, device=device, epoch=epoch, use_weights=use_class_weighting)
        scheduler.step()
        # 验证Validate
        val_loss, val_accuracy = val_one_epoch(model=model, data_loader=val_loader, device=device, epoch=epoch, use_weights=use_class_weighting)

        # 添加至tensorboard
        metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"]
        }
        
        for tag, value in metrics.items():
            tensorboard_writer.add_scalar(tag, value, epoch)

        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f"./model_weights/model_epoch_{epoch}.pth")

if __name__ == '__main__':
    opt = parse_args()
    run_training(opt)