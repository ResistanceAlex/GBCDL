import torch
import torch.optim as optim

from torch.cuda.amp import GradScaler, autocast

import timeit
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
import os

# 将目录加载到系统环境中，其中os.path.dirname()是获得当前文件的父目录
# 并将其加载到环境变量中 (注：这种环境变量只在运行时生效，程序运行结束即失效)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from GBCUtil.dataUtil import get_loaders
from GBCUtil.loss import FocalLoss
from GBCUtil.loss import TverskyLoss
from Model.model import ResNetChange
from Model.modelTest import Resnet101

# 检查是否有 GPU 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
定义评估函数,对给定的数据集进行评估

param:
- name: 评估数据集的名称
- model: 已训练的模型
- data_loader: DataLoader，提供评估数据
- device: 设备，'cuda' 或 'cpu'

return:
- accuracy: 预测准确率
'''
def evaluate_model(name, model, data_loader, device):

    # 将模型切换到评估模式
    model.eval()
    correct_predictions = 0
    total_samples = 0

    # 初始化进度条，用于显示评估进度
    progress_bar = tqdm(data_loader, desc=f"{name} evaluating", unit="batch")
    
    # 关闭梯度计算以提高速度
    with torch.no_grad():
        for inputs, labels in progress_bar:

            # 将数据移动到 GPU
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播：计算模型输出
            outputs = model(inputs)

            # 使用 torch.max 获取每个样本的预测类别
            _, predictions = torch.max(outputs, 1)

            # 累加正确预测的数量
            correct_predictions += (predictions == labels).sum().item()

            # 累加处理的样本总数
            total_samples += labels.size(0)

            # 更新进度条的后缀，显示当前准确率
            progress_bar.set_postfix(accuracy=100 * correct_predictions / total_samples)

    # 计算并返回这个 epoch 的总准确率
    accuracy = 100 * correct_predictions / total_samples

    return accuracy

'''
定义训练函数,对给定的数据集进行训练

param:
- model: 已训练的模型
- criterion: 损失函数
- optimizer: 优化器
- train_loader: DataLoader，提供训练数据
- valid_loader: DataLoader，提供验证数据
- scheduler: 调度器
- modelName: 保存的模型名称
- device: 设备，'cuda' 或 'cpu'

return:
- plt_data: 训练数据
'''
def train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs, scheduler, modelName, device):
    
    plt_data = {
        'loss': [],
        'train_accuracy': [],
        'valid_accuracy': []
    }
    
    # 初始化梯度缩放器
    scaler = GradScaler()
    
    for epoch in range(num_epochs):

        print(f'Epoch {epoch+1}/{num_epochs} Working:')

        # 记录训练开始时间
        start_train = timeit.default_timer()

        # 将模型切换到训练模式
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"training", unit="batch")
        
        for inputs, labels in progress_bar:

            # 将数据移动到 GPU
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 使用 autocast 上下文管理器来运行前向传播，使用混合精度
            with autocast():

                # 前向传播：计算模型对当前输入的输出
                outputs = model(inputs)

                # 计算损失函数
                loss = criterion(outputs, labels)

            # 反向传播前使用梯度缩放
            scaler.scale(loss).backward()

            # 使用缩放器来调用优化器的 step 函数
            scaler.step(optimizer)

            # 更新缩放器
            scaler.update()

            # 累加本批次的损失
            running_loss += loss.item()

            # 更新进度条的损失信息
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))
        
        # 计算并显示训练集和验证集的准确率
        train_accuracy = evaluate_model('Train', model, train_loader, device)
        valid_accuracy = evaluate_model('Vaild', model, valid_loader, device)

        # 保存计算的损失和准确率
        plt_data['loss'].append(running_loss / len(train_loader))
        plt_data['train_accuracy'].append(train_accuracy)
        plt_data['valid_accuracy'].append(valid_accuracy)

        stop_val = timeit.default_timer()  # 记录训练结束时间
        
        print("-" * 30)
        print(f"Epoch {epoch + 1} [Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {valid_accuracy:.4f}]")
       
        # 更新学习率
        scheduler.step(running_loss/len(train_loader))

        print(f"Epoch {epoch + 1} Training Time:{stop_val - start_train:.2f}s")
        print("-" * 30)
    
    # 保存训练后模型的状态
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': plt_data['loss'],
        'train_accuracy': plt_data['train_accuracy'],
        'valid_accuracy': plt_data['valid_accuracy']
    }, modelName)

    return plt_data

# 加载训练集、验证集和测试集数据
batch_size = 32
train_csv_dir = 'E:/pyDLW/GBCDL/data/train.csv'
val_csv_dir = 'E:/pyDLW/GBCDL/data/val.csv'
    
train_dataloader = get_loaders('train', train_csv_dir, batch_size, 512, 1)
val_dataloader = get_loaders('val', val_csv_dir, batch_size, 512, 1)

# 设定dropout
dropout = 0.3
model = ResNetChange(3, dropout)

# 载入保存的模型状态
checkpoint = torch.load('result/model/Resnetch_TverskyLoss.pth')
model.load_state_dict(checkpoint['model_state_dict'])


model.to(device)  # 将模型移动到 GPU 上

'''
Label Counts:
meCT: 9235
noCT: 8107
opCT: 4564
sum:21906
Label %:
meCT: 0.42157399799141787
noCT: 0.37008125627681915
opCT: 0.20834474573176298
'''

# 设定训练轮次
num_epochs = 50

# 定义模型、损失函数、优化器和调度器
# 损失函数：
criterion = TverskyLoss(alpha=0.2083, beta=0.7917)

# 优化器：AdamW
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# 调度器：OneCycleLR
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                max_lr=0.01, 
                                                total_steps=num_epochs, 
                                                steps_per_epoch=10, 
                                                verbose=True)

# 保存模型的名称
modelName = 'Resnet101_FocalLoss.pth'

start = timeit.default_timer()


print("-" * 30)
print("start")
print("-" * 30)

# 训练模型
plt_data = train_model(model, criterion, optimizer, train_dataloader, val_dataloader, num_epochs, scheduler, modelName, device)

stop = timeit.default_timer()
print(f"Total Training Time:{stop - start:.2f}s")