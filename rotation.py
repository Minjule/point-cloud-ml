import open3d as o3d
import numpy as np
import json 
import os

datas = os.listdir("pcd\\data\\")
augmented_data = os.listdir("pcd\\augmented\\data\\")
array = np.arange(0, 181, 10)
for file in datas:
    next = str(len(augmented_data))
    pcd = o3d.io.read_point_cloud("pcd\\data\\"+str(file))
    pcd_augmented = o3d.io.read_point_cloud("pcd\\data\\"+str(file))
    labelfile = open('pcd\\labels\\'+(str(file[:-4]))+".json")
    label = json.load(labelfile)

    degree = np.random.choice(array)
    theta = np.deg2rad(degree)  # Rotation angle in radians
    rotation_matrix = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    for i in label['objects']:
        center = np.asarray([i['centroid']["x"], i['centroid']["y"], i['centroid']["z"]])
        transformed_center = (center @ rotation_matrix.T)
        i['centroid']["x"] = transformed_center[0]
        i['centroid']["y"] = transformed_center[1]
        i['centroid']["z"] = transformed_center[2]
    
    
    points_np = np.asarray(pcd.points)
    transformed_points = (points_np @ rotation_matrix.T)

    pcd_augmented.points = o3d.utility.Vector3dVector(transformed_points)

    if pcd == pcd_augmented:
        print("same")

    next = f"{int(next):010}"
    name = '{:0>10}'.format("pcd\\augmented\\labels\\"+str(next)+".json")
    #o3d.visualization.draw_geometries([pcd])

    with open(str(name), "w") as outfile:
        json.dump(label, outfile, indent=4)

    o3d.io.write_point_cloud("pcd\\augmented\\data\\"+str(next)+".pcd", pcd_augmented)

    #print(name)
    augmented_data = os.listdir("pcd\\augmented\\data\\")