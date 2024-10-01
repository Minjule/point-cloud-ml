import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os 
import json

augmented_data = os.listdir("pcd\\augmented\\data\\")
centers = {}
centers['x'] = []
centers['y'] = []
centers['z'] = []

for file in augmented_data:
    label = json.load(open("pcd\\augmented\\labels\\" + str(file)[:-4] + ".json"))
    for i in label['objects']:
        centers['x'].append(i['centroid']["x"])
        centers['y'].append(i['centroid']["y"])
        centers['z'].append(i['centroid']["z"])

df_centers = pd.DataFrame(centers)

sns.scatterplot(x='z', y='y', data=df_centers)

plt.show()