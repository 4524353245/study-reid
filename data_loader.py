from __future__ import print_function,absolute_import
import os
from PIL import Image
import numpy as np
import os.path as osp

import torch
from torch.untils.data import Dataset


def read_img(img_path):
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("{} does not read image".format(img_path))
        pass
    return img

# 重构torch的dataset
class ImageDataset(Dataset):
    def __init__(self,dataset,transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,index):
        img_path, pid, camid = self.dataset[index]
        img = read_img(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img,pid,camid


        

img = Image.open(img_path).convert('RGB')



if __name__ == '__main__':
    import data_manager
    dataset = data_manager.init_img_dataset(root='C:\\Users\\surface\\Desktop')
    train_loader = ImageDataset(dataset.train)