# 使用pytorch中定义好的网络模型
import torch.nn as nn
# 导入自带的网络包
from torchvision import models
from einops.layers.torch import Rearrange

class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        # pretrained加载预训练模型,无预训练模型会自动下载
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_features, 3)
 
    def forward(self, x):
        out = self.model(x)
        return out

class Resnet101(nn.Module):
    def __init__(self):
        super(Resnet101, self).__init__()
        # pretrained加载预训练模型,无预训练模型会自动下载
        self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_features, 3)
 
    def forward(self, x):
        out = self.model(x)
        return out

class Resnet152(nn.Module):
    def __init__(self):
        super(Resnet152, self).__init__()
        # pretrained加载预训练模型,无预训练模型会自动下载
        self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_features, 3)
 
    def forward(self, x):
        out = self.model(x)
        return out
    
class Densenet121(nn.Module):
    def __init__(self):
        super(Densenet121, self).__init__()
        # pretrained加载预训练模型,无预训练模型会自动下载
        self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
 
    def forward(self, x):
        out = self.model(x)
        return out

class Densenet201(nn.Module):
    def __init__(self):
        super(Densenet201, self).__init__()
        # pretrained加载预训练模型,无预训练模型会自动下载
        self.model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
 
    def forward(self, x):
        out = self.model(x)
        return out


class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        self.model = models.vit_b_32(models.ViT_B_32_Weights.DEFAULT)
 
    def forward(self, x):
        out = self.model(x)
        return out

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # pretrained加载预训练模型,无预训练模型会自动下载
        self.model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_features, 3)
 
    def forward(self, x):
        out = self.model(x)
        return out
