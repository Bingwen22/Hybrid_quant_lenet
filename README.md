# Hrbrid Quantization LeNet for MNIST

## 项目简介

本项目实现了对LeNet模型进行混合量化（Hybrid Quantization）以应用于MNIST手写数字识别任务。通过混合量化技术，我们能够在保证模型性能的同时，减少模型的存储需求和计算开销，从而使得模型更易于在资源受限的设备上部署。

## 项目结构
* `/checkpoint`: 存放训练好的模型参数文件。
* `/data`: 存放数据集文件。
* `/models`: 存放模型定义文件。
* `/utils`: 存放数据处理、模型评估等辅助函数。
* `log`: 训练日志。
* `train.py`: 训练脚本。
* `eval.py`: 评估脚本。

## 环境配置
1. Python环境：推荐使用Python 3.7 及以上版本
2. 安装依赖包：使用pip命令安装以下依赖包：
```bash
    pip install -r requirements.txt
```

## 数据准备
1. 运行`train.py` 时会自动将数据集下载hip格式，并解压到`./data`目录下。

## 训练步骤
1. 运行`train.py` 脚本，测试结果将显示在终端中，相关训练参数同时保存在 `/log/{time}`下，其中 time 是开始训练的时间


