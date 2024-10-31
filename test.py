import argparse
import torch
from torchvision import transforms
from dataset import MyDataSet
from net import create_model
from utils import load_test_data
from thop import profile

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    batch_size = args.batch_size
    test_images_path, test_images_label = load_test_data(args.data_path)
    data_transform = transforms.Compose(
        [transforms.Resize(36),
         transforms.CenterCrop(32),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    test_dataset = MyDataSet(images_path=test_images_path,
                              images_class=test_images_label,
                              transform=data_transform)
    model = create_model(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load('./weights2/model-199.pth'))
    input_tensor = torch.randn(1,3,32,32).to(device)
    flops, params = profile(model, inputs=(input_tensor,))
    print(f"模型参数量：{params/1e6:.2f} M")
    print(f"模型FLOPs: {flops/1e9:.2f} G")
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            model.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'测试集上的准确率为: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--data-path', type=str,
                        default="./CIFAR10_balance")
    opt = parser.parse_args()
    main(opt)
