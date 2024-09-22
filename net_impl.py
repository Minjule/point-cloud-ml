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
train_loader = DataLoader(train_data, batch_size=2, shuffle=True)

if __name__ == '__main__':
    gpus = [0]
    device = torch.device(f'cuda:{gpus[0]}' if torch.cuda.is_available() else 'cpu')
    cnt = 0 
    print("Loading data ... ")
    for x, y in train_loader:
        x = x.to(device)
        kernel_size = 218283 // 5000
        stride = kernel_size
        avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)
        x = avg_pool(x)
        y = y.to(device)

        bs = x.shape[0]
        print(x[1][1])
        out = nn.Conv1d(3, 64, kernel_size=1)(x)
        out = (F.relu(out))
        out = nn.BatchNorm1d(64)(out)
        print(out[1][1])
        print(out.shape)

        out = nn.Conv1d(64, 128, kernel_size=1)(out)
        out = (F.relu(out))
        out = nn.BatchNorm1d(128)(out)
        print(out[1][1])
        print(out.shape)

        out = nn.Conv1d(128, 1024, kernel_size=1)(out)
        out = (F.relu(out))
        out = nn.BatchNorm1d(1024)(out)
        print(out[1][1])
        print(out.shape)

        out = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)(out).view(bs, -1)
        print(out[1][1])
        print(out.shape)
        print(out.size(0))
        out = out.view(out.size(0), -1)
        print(out[1][1])
        print(out.shape)

        cnt += 1
        if(cnt > 0):
            break