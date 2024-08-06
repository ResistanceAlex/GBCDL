import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

import sys
import os

# 将目录加载到系统环境中，其中os.path.dirname()是获得当前文件的父目录
# 并将其加载到环境变量中 (注：这种环境变量只在运行时生效，程序运行结束即失效)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from GBCUtil.GBCData import GBCTrainDataset, GBCTestDataset, GBCValDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
加载数据集函数

param：
- catetory: 类别名
- csv_dir: 数据的csv
- batch_size: 每批次数据的图片数量
- img_size: 图片大小
- subset_ratio: 数据集比例

return:
- dataloader: 数据集
'''
def get_loaders(catetory, csv_dir, batch_size, img_size, subset_ratio):
    if catetory == 'train':
        data_db = GBCTrainDataset(csv_dir, img_size)
    elif catetory == 'test':
        data_db = GBCTestDataset(csv_dir, img_size)
    elif catetory == 'val':
        data_db = GBCValDataset(csv_dir, img_size)
    else:
        print(f"catetory{catetory} is wrong ,plaese checkout your catetory")

    # 然后，如果需要，从加载的数据集中选择子集
    if subset_ratio < 1.0:
        subset_indices = np.random.choice(len(data_db), int(len(data_db) * subset_ratio), replace=False)
        data_db = Subset(data_db, subset_indices)

    # 创建DataLoader实例
    dataloader = DataLoader(dataset=data_db, batch_size=batch_size, shuffle=True)
    return dataloader

if __name__ == '__main__':
    batch_size = 32
    train_csv_dir = 'E:/GBCDL/data/train.csv'
    val_csv_dir = 'E:/GBCDL/data/val.csv'
    test_csv_dir = 'E:/GBCDL/data/test.csv'
    train_dataloader = get_loaders('train', train_csv_dir, batch_size, 512, 0.1)
    val_dataloader = get_loaders('val', val_csv_dir, batch_size, 512, 0.1)
    test_dataloader = get_loaders('test', test_csv_dir, batch_size, 512, 0.1)