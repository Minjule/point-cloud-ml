import torch
import os
import json
import open3d as o3d
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import json 

class PointNetDataset(Dataset):
  def __init__(self, root_dir):
    super(PointNetDataset, self).__init__()

    self._features = []
    self._boxes = []

    self.load_trainD(root_dir)

  def __len__(self):
    return len(self._features)
  
  def __getitem__(self, idx):
    feature, box = self._features[idx], self._boxes[idx]
    
    # normalize input to zero-mean
    feature = feature - np.mean(feature, axis=0, keepdims=True) # center; (N, 3)
    max_dist = np.max(np.linalg.norm(feature, axis=1))
    feature = feature / max_dist # scale

    # random rotation
    theta = np.random.uniform(0, np.pi * 2)
    # rotation_matrix = np.array([[np.cos(theta), 0, -np.sin(theta)],
    #                             [0, 1, 0],
    #                             [np.sin(theta), 0, np.cos(theta)]])
    # less computation
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    feature[:, [0, 2]] = feature[:, [0, 2]].dot(rotation_matrix)

    # random jitter
    feature += np.random.normal(0, 0.02, size=feature.shape)
    feature = torch.Tensor(feature.T)

    return feature, box
  
  def load_trainD(self, dir):
    pcds = os.listdir(str(dir) + "\\data")
    for pcd in pcds:
      points = o3d.io.read_point_cloud("pcd\\augmented\\data\\"+ str(pcd))
      points_np = np.asarray(points.points)
      points_np = np.full([218295, 3], np.nan)
      points_np = points_np[12:]

      self._features.append(points_np)
      label = json.load(open(str(dir) + "\\labels\\" + str(pcd)[:-4] + ".json"))
      for i in label['objects']:
        box_ = []
        box_.append(np.asarray([[i['centroid']["x"], i['centroid']["y"], i['centroid']["z"]],[i['dimensions']["length"], i['dimensions']["width"], i['dimensions']["height"]]]))
      self._boxes.append(box_)
      
    self._features = np.array(self._features)
    self._boxes = np.array(self._boxes)

if __name__ == "__main__":
  train_data = PointNetDataset("pcd\\augmented")
  print(len(train_data))
  train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
  cnt = 0
  for pts, label in train_loader:
    # print(pts.shape)
    print(label.shape)
    print(label)
    cnt += 1
    if cnt > 3:
      break