import open3d as o3d
import numpy as np
import json 
import os

datas = os.listdir("pcd\\data\\")
augmented_data = os.listdir("pcd\\augmented\\data\\")
array = np.arange(0, 181, 10)

pcd = o3d.io.read_point_cloud("pcd\\augmented\\data\\0000000000.pcd")
points_np = np.asarray(pcd.points)
points_np = points_np[12:]
max_x, max_y, max_z = np.nanmax(points_np, axis=0)
min_x, min_y, min_z = np.nanmin(points_np, axis=0) 

global_min = np.array([np.inf, np.inf, np.inf])
global_max = np.array([-np.inf, -np.inf, -np.inf])
max_len = 0
for file in augmented_data:
    pcd = o3d.io.read_point_cloud("pcd\\augmented\\data\\"+str(file))
    label = json.load(open("pcd\\augmented\\labels\\" + str(file)[:-4] + ".json"))
    pcd = np.asarray(pcd.points)
    valid_rows = ~np.all(np.isnan(pcd), axis=1)
    filtered_pcd = pcd[valid_rows]

    l_min = np.min(filtered_pcd, axis=0)
    l_max = np.max(filtered_pcd, axis=0)

    global_min = np.minimum(global_min, l_min)
    global_max = np.maximum(global_max, l_max)
    if(max_len < len(label['objects'])):
        max_len = len(label['objects'])

    pcd = np.full([218295, 3], np.nan)
    
print(global_min)
print(global_max)
print(max_len)