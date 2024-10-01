import open3d as o3d
import numpy as np
import json
import os

datas = os.listdir("pcd\\data\\")
augmented_data = os.listdir("pcd\\augmented\\data\\")
for file in datas:
    next = str(len(augmented_data))
    pcd = o3d.io.read_point_cloud("pcd\\data\\"+str(file))
    points = np.asarray(pcd.points)
    pcd_augmented = o3d.io.read_point_cloud("pcd\\data\\"+str(file))
    labelfile = open('pcd\\labels\\'+(str(file[:-4]))+".json")
    label = json.load(labelfile)
    #x = np.random.uniform(0, 2)   #it would be best if it almost not changed
    #y = np.random.uniform(-1, 1)  #it can not be changed
    z = np.random.uniform(-3.5, 0.5)  #it would be best if it is negative
    translation_vector = np.array([0.0, 0.0, z])

    for i in label['objects']:
        z = np.random.uniform(-3.5, 0.5)  #it would be best if it is negative
        translation_vector = np.array([0.0, 0.0, z])
        length = float((i['dimensions']["length"]/2)+0.025)
        width = float((i['dimensions']["width"]/2)+0.025)
        height = float(i['dimensions']["height"]/2)
        min_bound = np.array([i['centroid']["x"]-length, i['centroid']["y"]-width, i['centroid']["z"]-height])
        max_bound = np.array([i['centroid']["x"]+length, i['centroid']["y"]+width, i['centroid']["z"]+height])
        indices = np.where(
            (points[:, 0] >= min_bound[0]) & (points[:, 0] <= max_bound[0]) &
            (points[:, 1] >= min_bound[1]) & (points[:, 1] <= max_bound[1]) &
            (points[:, 2] >= min_bound[2]) & (points[:, 2] <= max_bound[2])
        )[0]

        points[indices] += translation_vector
        pcd.points = o3d.utility.Vector3dVector(points)
        #pcd_augmented.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
       # pcd_augmented.points[indices] = o3d.utility.Vector3dVector(np.asarray(selected_points))
        i['centroid']["x"] = i['centroid']["x"] + translation_vector[0]
        i['centroid']["y"] = i['centroid']["y"] + translation_vector[1]
        i['centroid']["z"] = i['centroid']["z"] + translation_vector[2]

    next = f"{int(next):010}"
    name = '{:0>10}'.format("pcd\\augmented\\labels\\"+str(next)+".json")
    #o3d.visualization.draw_geometries([pcd])

    with open(str(name), "w") as outfile:
        json.dump(label, outfile, indent=4)

    o3d.io.write_point_cloud("pcd\\augmented\\data\\"+str(next)+".pcd", pcd)

    #print(name)
    augmented_data = os.listdir("pcd\\augmented\\data\\")