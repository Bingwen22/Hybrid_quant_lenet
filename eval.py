from PIL import Image
from mpl_toolkits.mplot3d.proj3d import transform
from mpmath.identification import transforms

from models.lenet import *
from utils.tools import *


def eval_dataset():
    _, test_loader = get_dataloaders()
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')


def eval_single_img(img_path):
    transform = transforms.Compose([transforms.ToTensor()])
    img = Image.open(img_path)
    img = transform(img)        # 转换成 Tensor 类型
    img = img.unsqueeze(0).to(device)   # 增加batch维度

    output = model(img)
    _, predicted = torch.max(output, 1)
    print(f"Image({img_path}) is predicted as {predicted}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet_Quant(quant_cfg=[4,4,4,4,4])
    model.load_state_dict(torch.load('./checkpoint/2024_10_31_23_00/4_4_4_4_4_best.pth', weights_only=True))
    model.to(device)
    model.eval()

    eval_dataset()
    eval_single_img('./data/test_imgs/image_0_label_7.png')

