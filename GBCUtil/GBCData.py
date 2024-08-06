import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

'''
class GBCTrainDataset
'''
class GBCTrainDataset(Dataset):
    def __init__(self, csv_file, resize):
        self.data = pd.read_csv(csv_file)
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            # 随机旋转进行数据增强
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label

class GBCTestDataset(Dataset):
    def __init__(self, csv_file, resize):
        self.data = pd.read_csv(csv_file)
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label
    
class GBCValDataset(Dataset):
    def __init__(self, csv_file, resize):
        self.data = pd.read_csv(csv_file)
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label