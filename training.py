import os
import re
from glob import glob
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchmetrics.classification import MulticlassMatthewsCorrCoef
import open3d as o3d

import matplotlib as mpl
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torch.optim as optim
from loss import PointNetLoss

root = r'C:\\Users\\Acer\\Documents\\GitHub\\point-cloud-ml\\pcd'

GLOBAL_FEATS = 1024
BATCH_SIZE = 32
