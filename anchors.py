import numpy as np
from math import sqrt as sqrt
from itertools import product as product

infos = {
    'grid_size': (10, 10, 10),  # Number of grid points in (x, y, z) dimensions
    'min_sizes': [0.13, 0.13, 0.13],  # Min sizes for anchor boxes in 3D
    'max_sizes': [0.18, 0.18, 0.18],  # Max sizes for anchor boxes in 3D
    'aspect_ratios': [(1, 1, 1), (1.5, 1, 1), (1, 1.5, 1)],  # Aspect ratios for anchor boxes
    'steps': [0.5, 0.5, 0.5],  # Step size for the anchor grid in 3D space
    'point_cloud_range': [-8.90265846,  -8.36585426, -10.08037663, 10.86396313,  5.29646969, 14.27643776],  # 3D space range in (x_min, y_min, z_min, x_max, y_max, z_max)
    'clip': True  
}

class AnchorBox(object):
  def __init__(self, config):
    super(AnchorBox, self).__init__()
    
    self.grid_size = config['grid_size']
    self.point_cloud_range = config['point_cloud_range']
    self.min_sizes = config['min_sizes']
    self.max_sizes = config['max_sizes']
    self.steps = config['steps']
    self.aspect_ratios = config['aspect_ratios']
    self.clip = config['clip']

  def __call__(self):
    boxes = []
    x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
    
    step_x = (x_max - x_min) / self.grid_size[0]
    step_y = (y_max - y_min) / self.grid_size[1]
    step_z = (z_max - z_min) / self.grid_size[2]

    x_centers = np.arange(x_min + step_x / 2, x_max, step_x)
    y_centers = np.arange(y_min + step_y / 2, y_max, step_y)
    z_centers = np.arange(z_min + step_z / 2, z_max, step_z)
    
    for cx, cy, cz in product(x_centers, y_centers, z_centers):
      for min_size, max_size in zip(self.min_sizes, self.max_sizes):
        s_k = min_size
        s_k_prime = sqrt(min_size * max_size)
        boxes.append([cx, cy, cz, s_k, s_k, s_k])
        boxes.append([cx, cy, cz, s_k_prime, s_k_prime, s_k_prime])

        for ratio in self.aspect_ratios:
            w = s_k * sqrt(ratio[0])
            h = s_k * sqrt(ratio[1])
            d = s_k * sqrt(ratio[2])
            boxes.append([cx, cy, cz, w, h, d])
            boxes.append([cx, cy, cz, h, w, d])

    output = np.array(boxes).reshape([-1, 6])  # shape: [8732, 4]
    if self.clip:
      output = np.clip(output, a_min=-10, a_max=14)
    return output


if __name__ == '__main__':
  boxes = AnchorBox(infos)()
  print(boxes)
  print(boxes.shape)