import torch
from tqdm import tqdm
import torchvision.models as models

import sys
import os

# 将目录加载到系统环境中，其中os.path.dirname()是获得当前文件的父目录
# 并将其加载到环境变量中 (注：这种环境变量只在运行时生效，程序运行结束即失效)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from GBCUtil.dataUtil import get_loaders

from Model.modelTest import Resnet101
from Model.model import ResNetChange


'''
定义加载模型的函数,加载指定路径的模型

param：
- model_path: 字符串，保存模型状态字典的路径
- num_classes: 整数，模型输出的类别数量

return:
- model: 加载了预训练权重的模型
'''
def load_model(model_path, num_classes):

    # 设定dropout
    dropout = 0.3
    model = ResNetChange(3, dropout)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)

    # 加载训练好的模型参数
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


'''
定义预测函数,对给定的数据集进行预测

param:
- model: 已训练的模型
- data_loader: DataLoader，提供测试数据
- device: 设备，'cuda' 或 'cpu'

return:
- predictions: 预测结果列表。
'''
def predict(model, data_loader, device):
    
    # 将模型切换到评估模式
    model.eval()
    correct_predictions = 0
    total_samples = 0

    # 初始化进度条，用于显示评估进度
    progress_bar = tqdm(data_loader, desc="predicting", unit="batch")
    
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

# 主函数
if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 64
    test_csv_dir = 'E:/GBCDL/data/test.csv'
    test_dataloader = get_loaders('test', test_csv_dir, batch_size, 512, 1)
    
    # 创建模型实例
    # 设定dropout
    dropout = 0.3
    model = ResNetChange(3, dropout).to(device)

    # 载入保存的模型状态
    checkpoint = torch.load('result/model/Resnetch_TverskyLoss.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # 使用模型进行预测
    predictions = predict(model, test_dataloader, device)

    # 打印预测结果
    print("Predictions:", predictions)

