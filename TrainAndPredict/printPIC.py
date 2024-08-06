import torch
import numpy as np 
import matplotlib.pyplot as plt

# 绘制训练结果
'''
绘制训练结果

param:
- plt_data: 训练数据

return:
- none
'''
def plot_training_results(plt_data):

    # 正常显示中文
    plt.rcParams['font.family'] = 'SimHei'

    # 正常显示负号
    plt.rcParams['axes.unicode_minus'] = False

    # 绘制训练损失
    plt.figure(figsize=(10, 5))
    plt.plot(plt_data['loss'], label='Training Loss', color='tab:red', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()

    # 绘制训练精度, 验证精度, 并在指定轮次上绘制测试精度
    plt.figure(figsize=(10, 5))

    # 绘制训练精度和验证精度
    plt.plot(plt_data['train_accuracy'], label='Training Accuracy', color='tab:blue', marker='o')
    plt.plot(plt_data['valid_accuracy'], label='Validation Accuracy', color='tab:green', marker='o')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()

# 主函数
if __name__ == '__main__':

    # 载入保存的模型状态
    checkpoint = torch.load('result/model/Resnetch_TverskyLoss.pth')
    plt_data = {
        'loss': [],
        'train_accuracy': [],
        'valid_accuracy': []
    }

    plt_data['loss'] = checkpoint['loss']
    plt_data['train_accuracy'] = checkpoint['train_accuracy']
    plt_data['valid_accuracy'] = checkpoint['valid_accuracy']
    

    # 调用训练函数并绘图
    plot_training_results(plt_data)
