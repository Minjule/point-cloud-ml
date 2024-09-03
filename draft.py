import open3d as o3d
import numpy as np
import json 
import os

datas = os.listdir("pcd\\data\\")
augmented_data = os.listdir("pcd\\augmented\\data\\")
array = np.arange(0, 181, 10)
max = 0
for file in augmented_data:
    pcd = o3d.io.read_point_cloud("pcd\\augmented\\data\\"+str(file))
    pcd = np.asarray(pcd.points)
    pcd = np.full([218295, 3], np.nan)
    print(len(pcd))