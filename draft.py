import open3d as o3d
import numpy as np
import json 
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from point_net import *
from torch.utils.data import DataLoader
from dataset import PointNetDataset

train_data = PointNetDataset("C:\\Users\\Acer\\Documents\\GitHub\\point-cloud-ml\\pcd\\augmented")
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

if __name__ == '__main__':
    gpus = [0]
    device = torch.device(f'cuda:{gpus[0]}' if torch.cuda.is_available() else 'cpu')
    detect = PointNetDetectHead().to(device=device)

    for x, y in train_loader:
        x = x.to(device)
        kernel_size = 218283 // 5000
        stride = kernel_size
        avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)
        x = avg_pool(x)

        print(x.shape)
        y = y.to(device)

        out = detect(x)

        print(out)
