import torch
import torch.nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4)

    return train_loader, test_loader


# 保存图片的函数
def save_images(save_dir, num_images=10):
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 加载 MNIST 数据集
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(num_images):
        image, label = test_dataset[i]
        image = transforms.ToPILImage()(image)  # 将张量转换为 PIL 图像
        image_path = os.path.join(save_dir, f'image_{i}_label_{label}.png')
        image.save(image_path)
        print(f'Saved {image_path}')


def save_params(model, save_dir):
    pass

if __name__ == '__main__':
    save_images('../data/test_imgs', num_images=10)