import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from models.lenet import LeNet, LeNet_Quant
from utils.tools import get_dataloaders
from tqdm import tqdm
from datetime import datetime

def train(model, epoch, optimizer):
    model.train()
    with tqdm(total=len(train_loader), desc=f'Train Epoch {epoch}') as pbar:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
            pbar.update(1)


@torch.no_grad()
def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
    return accuracy


def mixed_quantization_train(model, best_model_path):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    best_acc = 0.0
    for epoch in range(5):
        train(model, epoch, optimizer)
        cur_acc = test(model)
        logging.info(f'Epoch {epoch + 1}: Test Accuracy: {cur_acc:.2f}%')

        if best_acc < cur_acc:
            best_acc = cur_acc
            torch.save(model.state_dict(), best_model_path)
            print('Saved best model!')
    return best_acc


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_dataloaders()
    criterion = nn.CrossEntropyLoss()

    # Set Path
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
    log_file = f'./log/{timestamp}_train.log'
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print('Training LeNet-5 with Quantization')
    quant_cfg = [8, 8, 8, 8, 8] # initial quantization config

    log_message = "本次训练量化参数：\n"
    for j, bit in enumerate(quant_cfg):
        log_message += f"Layer #{j}: Weights and Activations are quantized to {bit}-bit\n"
    print(log_message)
    logging.info(log_message)

    os.makedirs(f'./checkpoint/{timestamp}', exist_ok=True)
    model = LeNet_Quant(quant_cfg=quant_cfg).to(device)
    best_model_path = f'./checkpoint/{timestamp}/best_tmp.pth'

    _ = mixed_quantization_train(model, best_model_path)

    while(1):
        best_quant_acc = 0.0

        # Search Best Quantization config
        for (idx, val) in enumerate(quant_cfg):
            if val == 4:
                continue
            layer_quant_cfg = quant_cfg.copy()
            layer_quant_cfg[idx] = 4
            log_message = f"Quantizing layer {idx + 1} to 4-bit\n"
            log_message += "本次训练量化参数：\n"
            for j, bit in enumerate(layer_quant_cfg):
                log_message += f"Layer #{j}: Weights and Activations are quantized to {bit}-bit\n"
            print(log_message)
            logging.info(log_message)

            quant_cfg_str = '_'.join([str(x) for x in layer_quant_cfg])

            model = LeNet_Quant(quant_cfg=layer_quant_cfg).to(device)
            if best_model_path is not None:
                model.load_state_dict(torch.load(best_model_path), strict=False)


            cur_model_path = f'./checkpoint/{timestamp}/{quant_cfg_str}_best.pth'
            cur_quant_acc = mixed_quantization_train(model, cur_model_path)

            if best_quant_acc < cur_quant_acc:
                best_quant_acc = cur_quant_acc
                best_quant_cfg = layer_quant_cfg.copy()
                best_model_path = cur_model_path

        quant_cfg = best_quant_cfg.copy()
        if all(x == 4 for x in quant_cfg):
                # finish training
            break