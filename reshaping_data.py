import open3d as o3d
import numpy as np
import os
import math
import matplotlib.pyplot as plt

datas = os.listdir("pcd\\augmented\\data\\")
width = 560  # Number of points along the x-axis

for file in datas:
    pcd = o3d.io.read_point_cloud("pcd\\augmented\\data\\"+str(file))
    points = np.asarray(pcd.points)

    sorted_indices = np.lexsort((points[:, 1], points[:, 0]))   #will sort rows by first column and then second
    points_np_sorted  = points[sorted_indices]
    num_points= len(points_np_sorted)
    height = math.ceil(num_points / width)

    total_required_points = width * height
    
    if num_points < total_required_points:
        #Pad the point cloud with zeros
        padding = np.zeros((total_required_points - num_points, points_np_sorted.shape[1]))
        points_padded = np.vstack((points, padding))
    else:
        #Crop the point clouds to fit the grid size
        points_padded = points_np_sorted[:total_required_points]

    
    organized_points = points_padded.reshape(height, width, points_np_sorted.shape[1])
    print(organized_points[2])
    #np.save("pcd\\organized\\data\\"+ str(file)[:-4]+".npy", organized_points)

# points = organized_points.reshape(-1, 3)

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)

#o3d.visualization.draw_geometries([pcd])