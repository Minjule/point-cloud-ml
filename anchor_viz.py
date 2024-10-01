from anchors import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import numpy as np

infos = {
    'grid_size': (10, 10, 10),  # Number of grid points in (x, y, z) dimensions
    'min_sizes': [0.13, 0.13, 0.13],  # Min sizes for anchor boxes in 3D
    'max_sizes': [0.18, 0.18, 0.18],  # Max sizes for anchor boxes in 3D
    'aspect_ratios': [(1, 1, 1), (1, 1.3, 1.5), (1, 1.5, 1.2)],  # Aspect ratios for anchor boxes
    'point_cloud_range': [-2.5,  -2.5, -2.5, 5,  2.5, 4.5],
    #'point_cloud_range': [-2.90265846,  0.13, 0.8, 0.18,  0.5, 1.5],  # 3D space range in (x_min, y_min, z_min, x_max, y_max, z_max)
    'clip': True 
}

anchors = AnchorBox(infos)()

y = [[[-0.6001,  0.0606, -0.4570], [ 0.1500,  0.1500,  0.1200]],
     [[-0.3913,  0.0712,  2.7192], [ 0.1800,  0.1500,  0.0900]],
     [[-0.6160,  0.0290,  1.9812], [ 0.1500,  0.1500,  0.1500]],
     [[-0.7963,  0.0072,  0.9155], [ 0.1500,  0.1500,  0.1500]],
     [[-0.9233,  0.0164, -0.1816], [ 0.1500,  0.1500,  0.1500]],
     [[ 0.4331,  0.2252,  0.2284], [ 0.1200,  0.1500,  0.1500]]]

def draw_3d_box(ax, y):
    vertices = []
    for b in range(len(y)):
        x_max, x_min = (y[0][0] + y[1][0]/2) , (y[0][0] - y[1][0]/2)
        y_max, y_min = (y[0][1] + y[1][1]/2) , (y[0][1] - y[1][1]/2)
        z_max, z_min = (y[0][2] + y[1][2]/2) , (y[0][2] - y[1][2]/2)
        vertices.append([x_min, y_min, z_min]) 
        vertices.append([x_max, y_max, z_max])

    vertices = np.array(vertices)
    edges = [[vertices[j] for j in [0, 1, 3, 2, 0]],  # bottom face
             [vertices[j] for j in [4, 5, 7, 6, 4]],  # top face
             [vertices[j] for j in [0, 4]],  # vertical edges
             [vertices[j] for j in [1, 5]],
             [vertices[j] for j in [2, 6]],
             [vertices[j] for j in [3, 7]]]
    
    for edge in edges:
        ax.plot3D(*zip(*edge), color='r')
    
if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Set limits and labels
    for b in y:
        draw_3d_box(ax, b)

    ax.set_xlim([0, 4])
    ax.set_ylim([0, 4])
    ax.set_zlim([0, 4])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()