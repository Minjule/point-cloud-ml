import open3d as o3d
import numpy as np
import json
import os

datas = os.listdir("pcd\\data\\")
augmented_data = os.listdir("pcd\\augmented\\data\\")
for file in datas:
    next = str(len(augmented_data))
    pcd = o3d.io.read_point_cloud("pcd\\data\\"+str(file))
    pcd_augmented = o3d.io.read_point_cloud("pcd\\data\\"+str(file))
    labelfile = open('pcd\\labels\\'+(str(file[:-4]))+".json")
    label = json.load(labelfile)
    translation_vector = np.random.uniform(-5, 5, size=3).astype(np.float32)

    for i in label['objects']:
        i['centroid']["x"] = i['centroid']["x"] + translation_vector[0]
        i['centroid']["y"] = i['centroid']["y"] + translation_vector[1]
        i['centroid']["z"] = i['centroid']["z"] + translation_vector[2]
    
    pcd_augmented.translate(translation_vector)
    if (pcd == pcd_augmented):
        print("not augmented")

    next = f"{int(next):010}"
    name = '{:0>10}'.format("pcd\\augmented\\labels\\"+str(next)+".json")

    with open(str(name), "w") as outfile:
         json.dump(label, outfile, indent=4)

    o3d.io.write_point_cloud("pcd\\augmented\\data\\"+str(next)+".pcd", pcd_augmented)

    print(name)
    augmented_data = os.listdir("pcd\\augmented\\data\\")