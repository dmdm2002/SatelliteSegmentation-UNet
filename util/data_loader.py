import glob
import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# root=D:/Side/AI_FACTORY/add
class CustomDataset(Dataset):
    def __init__(self, root: str, train: bool, dbs: list, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform

        self.img_path = []
        self.gt_path = []

        self.gt_values = [10, 30, 90, 70, 100]
        # self.gt_values = {
        #     10: 0,
        #     30: 1,
        #     90: 2,
        #     70: 3,
        #     100: 4
        # }
        if train:
            for db in dbs:
                img_folder = f'{root}/{db}/DB/Training/data/TS_Satellite_FGT_512pixel'
                gt_folder = f'{root}/{db}/DB/Training/label/TL_Satellite_FGT_512pixel/Tif'

                self.img_path += glob.glob(f'{img_folder}/*.tif')
                self.gt_path += glob.glob(f'{gt_folder}/*.tif')
        else:
            for db in dbs:
                img_folder = f'{root}/{db}/DB/Validation/data/VS_Satellite_FGT_512pixel'
                gt_folder = f'{root}/{db}/DB/Validation/label/VL_Satellite_FGT_512pixel/Tif'

                self.img_path += glob.glob(f'{img_folder}/*.tif')
                self.gt_path += glob.glob(f'{gt_folder}/*.tif')

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(self.gt_path[idx])
        # print(gt)
        # gt = np.vectorize(self.gt_values.get)(gt)
        gt = self.one_hot_encode(gt).astype('float')

        if self.transform is not None:
            augmented = self.transform(image=img, mask=gt)
            img = augmented['image']
            gt = augmented['mask']

        img = img.transpose(2, 0, 1)
        gt = gt.transpose(2, 0, 1)

        return torch.from_numpy(img), torch.from_numpy(gt)

    # 건물, 도로, 농경지, 산림, ignore --> 10 30 90 70 100
    def one_hot_encode(self, gt):
        semantic_map = []
        for colour in self.gt_values:
            equality = np.equal(gt, colour)
            class_map = np.all(equality, axis=-1)
            semantic_map.append(class_map)
        semantic_map = np.stack(semantic_map, axis=-1)

        return semantic_map
