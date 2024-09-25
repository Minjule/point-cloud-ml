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
    feature = np.nan_to_num(feature, nan=0.0)
    feature = feature.T
    feature = torch.tensor(feature, dtype=torch.float32)
    print(box)
    return feature, box
  
  def load_trainD(self, dir):
    pcds = os.listdir(str(dir) + "\\data")
    for pcd in pcds:
      print(f"\r{pcd}", end=" ", flush=True)
      points = o3d.io.read_point_cloud("pcd\\augmented\\data\\"+ str(pcd))
      points_np = np.asarray(points.points)
      points_np = (points_np - np.mean(points_np, axis=0)) / np.std(points_np, axis=0)

      pcd_added = np.zeros((218295, 3), dtype=float)
      pcd_added[:len(points_np)] = points_np
      pcd_added = pcd_added[12:]

      self._features.append(pcd_added)
      label = json.load(open(str(dir) + "\\labels\\" + str(pcd)[:-4] + ".json"))

      box_ = []
      for i in label['objects']:
        box_.append(np.asarray([[i['centroid']["x"], i['centroid']["y"], i['centroid']["z"]],[i['dimensions']["length"], i['dimensions']["width"], i['dimensions']["height"]]]))
      
      if len(box_) == 0:
        box_ = [np.full((2, 3), np.nan) for _ in range(6)]
      else:
        nan_arrays = [np.full(box_[0].shape, np.nan) for _ in range(6-len(box_))]
        if len(nan_arrays) != 0:
          box_ = np.concatenate([box_] + [nan_arrays], axis=0)
      self._boxes.append(box_)

    self._features = np.array(self._features)
    self._boxes = np.array(self._boxes)

if __name__ == "__main__":
  train_data = PointNetDataset("pcd\\augmented")
  print(len(train_data))
  train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
  cnt = 0
  for pts, label in train_loader:
    # print(pts.shape)
    print(label.shape)
    print(label)
    cnt += 1
    if cnt > 3:
      break