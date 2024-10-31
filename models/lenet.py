import torch
import torch.nn as nn
from utils.dorefa import *

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()

        # Define Model Architecture
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 16*4*4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class LeNet_Quant(nn.Module):
    def __init__(self, num_classes=10, quant_cfg=None):
        super(LeNet_Quant, self).__init__()
        if quant_cfg is None:
            quant_cfg = [8, 8, 8, 8, 8]

        self.conv1 = conv2d_Q_fn(quant_cfg[0], quant_cfg[0])(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = conv2d_Q_fn(quant_cfg[1], quant_cfg[1])(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = linear_Q_fn(quant_cfg[2], quant_cfg[2])(16*4*4, 120)
        self.fc2 = linear_Q_fn(quant_cfg[3], quant_cfg[3])(120, 84)
        self.fc3 = linear_Q_fn(quant_cfg[4], quant_cfg[4])(84, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 16*4*4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    model = LeNet()
    print(model)
    x = torch.randn(64, 1, 32, 32)
    out = model(x)
    print(out.size())