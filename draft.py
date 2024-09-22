import open3d as o3d
import numpy as np
import json 
import os

datas = os.listdir("pcd\\data\\")
augmented_data = os.listdir("pcd\\augmented\\data\\")
cnt = 0

for file in augmented_data:
    cnt += 1
    pcd = o3d.io.read_point_cloud("pcd\\augmented\\data\\"+str(file))
    label = json.load(open("pcd\\augmented\\labels\\" + str(file)[:-4] + ".json"))
      
    box_ = []
    _boxes = []
    for i in label['objects']:
        box_.append(np.asarray([[i['centroid']["x"], i['centroid']["y"], i['centroid']["z"]],[i['dimensions']["length"], i['dimensions']["width"], i['dimensions']["height"]]]))
    nan_arrays = [np.full(box_[0].shape, np.nan) for _ in range(6-len(box_))]
    print(len(box_))
    print(len(nan_arrays))
    box_ = np.concatenate([box_] + [nan_arrays], axis=0)
    _boxes.append(box_)
    print(len(box_))
    
    if (cnt > 0):
        break
