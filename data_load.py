import torch
import torchvision
import os
import pandas as pd
import torchvision.transforms as transforms
import numpy as np


#########################加载数据##############################
#数据增强
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
])

def read_data(is_train=True):
    """读取检测数据集中的图像和标签"""
    data_dir = 'detection/'
    csv_fname = os.path.join(data_dir, 'sysu_train' if is_train
                             else 'sysu_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    img_expend = []
    for img_name, target in csv_data.iterrows():
        img_expend.append(torchvision.io.read_image(
            os.path.join(data_dir, 'sysu_train' if is_train else
            'sysu_val', 'images', f'{img_name}')))
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'sysu_train' if is_train else
                         'sysu_val', 'images', f'{img_name}')))
        targets.append(list(target))
    print(img_expend[0].shape)
    for img_name, target in csv_data.iterrows():
        targets.append(list(target))
        
#数据扩充至原来的两倍
    for index in img_expend:
        index = data_transform(index)
        index = np.array(index)
        index = index.astype(np.float32)
        index = index.transpose(2, 0, 1)
        index = torch.tensor(index)
        images.append(index)
        # 这里的target包含（类别，左上角x，左上角y，右下角x，右下角y），
        # 其中所有图像都具有相同的类（索引为0）

    return images, torch.tensor(targets).unsqueeze(1) / 256

class Dataset(torch.utils.data.Dataset):
    """一个用于加载检测数据集的自定义数据集"""
    def __init__(self, is_train):
        self.features, self.labels = read_data(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)

def load_data(batch_size):
    """加载检测数据集"""
    train_iter = torch.utils.data.DataLoader(Dataset(is_train=True),
                                             batch_size, shuffle=True)
    #val_iter = torch.utils.data.DataLoader(Dataset(is_train=False),
    #                                       batch_size)
    return train_iter#, val_iter



