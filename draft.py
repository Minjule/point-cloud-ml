import open3d as o3d
import numpy as np
import json 
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from point_net import *

datas = os.listdir("pcd\\data\\")
augmented_data = os.listdir("pcd\\augmented\\data\\")
array = np.arange(0, 181, 10)
max = 0
for file in augmented_data:
    pcd = o3d.io.read_point_cloud("pcd\\augmented\\data\\"+str(file))
    pcd = np.asarray(pcd.points)
    pcd = np.full([218295, 3], np.nan)
    print(len(pcd))

tnet = Tnet(dim=3)
pointfeat = PointNetBackbone(local_feat=True)
detect = PointNetDetectHead(m=3)

# Define model, optimizer, and other components
model = MultiScalePointNetSSD(num_classes=3)  # Example with 3 object classes
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(dataloader, model, optimizer, num_epochs, num_classes):
    model.train()  # Set the model to training mode
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (point_clouds, gt_boxes, gt_labels) in enumerate(dataloader):
            optimizer.zero_grad()  # Zero the parameter gradients
            
            # Forward pass
            loc_preds, cls_preds = model(point_clouds)  # Predict bounding boxes and class scores
            
            # Assume `anchors` are predefined 3D anchors at various scales and positions
            anchors = generate_anchors()  # You need a function to define these
            
            # Match predicted boxes to ground truth
            matched_boxes = match_anchors_to_ground_truth(anchors, loc_preds, gt_boxes)
            
            # Compute loss (SSD loss function)
            loss = ssd_loss(loc_preds, cls_preds, matched_boxes, gt_labels, num_classes)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}")
