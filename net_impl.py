import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from point_net import *
from torch.utils.data import DataLoader
from dataset import PointNetDataset
from anchors import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

centers = {}

infos = {
    'grid_size': (10, 10, 10),  # Number of grid points in (x, y, z) dimensions
    'min_sizes': [0.4, 0.4, 0.4],  # Min sizes for anchor boxes in 3D
    'max_sizes': [0.6, 0.6, 0.6],  # Max sizes for anchor boxes in 3D
    'aspect_ratios': [(1, 1, 1), (1, 1.3, 1.5), (1, 1.5, 1.2)],  # Aspect ratios for anchor boxes
    'point_cloud_range': [-2.5,  -2.5, -2.5, 5,  2.5, 4.5],
    #'point_cloud_range': [-2.90265846,  0.13, 0.8, 0.18,  0.5, 1.5],  # 3D space range in (x_min, y_min, z_min, x_max, y_max, z_max)
    'clip': True 
}

train_data = PointNetDataset("C:\\Users\\Acer\\Documents\\GitHub\\point-cloud-ml\\pcd\\augmented")
train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    
def compute_intersection_volume(box1, box2):
        # Unpack box 1 and box 2
        center_x1, center_y1, center_z1, w1, h1, d1 = box1
        center_x2, center_y2, center_z2 = box2[0]
        w2, h2, d2 = box2[1]
        
        # Calculate min/max boundaries for each box
        x1_min, x1_max = center_x1 - w1 / 2, center_x1 + w1 / 2
        y1_min, y1_max = center_y1 - h1 / 2, center_y1 + h1 / 2
        z1_min, z1_max = center_z1 - d1 / 2, center_z1 + d1 / 2

        x2_min, x2_max = center_x2 - w2 / 2, center_x2 + w2 / 2
        y2_min, y2_max = center_y2 - h2 / 2, center_y2 + h2 / 2
        z2_min, z2_max = center_z2 - d2 / 2, center_z2 + d2 / 2
        
        x1_min, y1_min, z1_min = torch.tensor(x1_min), torch.tensor(y1_min), torch.tensor(z1_min)
        x1_max, y1_max, z1_max = torch.tensor(x1_max), torch.tensor(y1_max), torch.tensor(z1_max)

        # Calculate the overlap in each dimension
        x_overlap = torch.max(torch.zeros_like(x1_max), torch.min(x1_max, x2_max) - torch.max(x1_min, x2_min))
        y_overlap = torch.max(torch.zeros_like(y1_max), torch.min(y1_max, y2_max) - torch.max(y1_min, y2_min))
        z_overlap = torch.max(torch.zeros_like(z1_max), torch.min(z1_max, z2_max) - torch.max(z1_min, z2_min))
        
        # Compute intersection volume
        intersection_volume = x_overlap * y_overlap * z_overlap
        # print(f"x1_min - {x1_min}, y1_min - {y1_min}, z1_min - {z1_min}")
        # print(f"x1_max - {x1_max}, y1_max - {y1_max}, z1_max - {z1_max}")
        # print(f"x2_min - {x2_min}, y2_min - {y2_min}, z2_min - {z2_min}")
        # print(f"x2_max - {x2_max}, y2_max - {y2_max}, z2_max - {z2_max}")
        # print(f"overlaps --------------------------------------------- {torch.min(x1_max, x2_max)}, {torch.max(x1_min, x2_min)}, {torch.min(x1_max, x2_max)-torch.max(x1_min, x2_min)}")
        # print(f"intersection volume = {intersection_volume}")
        return intersection_volume
    
def compute_iou(box1, box2):   
        # This is a simplified IoU function. You need to define the exact intersection
        # volume calculation for 3D bounding boxes depending on your coordinate format.
        intersection_vol = compute_intersection_volume(box1, box2)  # You need this function
        vol_box1 = box1[3] * box1[4] * box1[5]  # width * height * depth
        vol_box2 = box2[1][0] * box2[1][1] * box2[1][2]
        union_vol = vol_box1 + vol_box2 - intersection_vol
        return intersection_vol / union_vol

def ssd_loss(loc_preds, matched_boxes):

        pos_mask = matched_boxes != None  # Positive samples mask
        
        # 1. Localization loss (Smooth L1 loss between matched GT boxes and predictions)
        loc_loss = F.smooth_l1_loss(loc_preds[pos_mask], matched_boxes[pos_mask], reduction='sum')
        
        return loc_loss

def non_maximum_suppression(pred_boxes, cls_scores, iou_threshold=0.5):
        indices = cls_scores.argsort(descending=True)  # Sort in descending order
        keep_boxes = []
        while len(indices) > 0:
            current_idx = indices[0]
            keep_boxes.append(pred_boxes[current_idx])
            remaining_boxes = pred_boxes[indices[1:]]
            ious = [compute_iou(pred_boxes[current_idx], box) for box in remaining_boxes]
            indices = indices[1:][[iou < iou_threshold for iou in ious]]
        return keep_boxes

if __name__ == '__main__':
    gpus = [0]
    device = torch.device(f'cuda:{gpus[0]}' if torch.cuda.is_available() else 'cpu')
    cnt = 0 
    print("Loading data ... ")
    for x, y in train_loader:
        x = x.to(device)
        kernel_size = 218283 // 5000
        stride = kernel_size
        avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)
        x = avg_pool(x)
        data = x
        y = y.to(device)
        print(y.shape)

        bs = x.shape[0]
        out = nn.Conv1d(3, 64, kernel_size=1)(x)
        out = (F.relu(out))
        out = nn.BatchNorm1d(64)(out)

        out = nn.Conv1d(64, 128, kernel_size=1)(out)
        out = (F.relu(out))
        out = nn.BatchNorm1d(128)(out)

        out = nn.Conv1d(128, 1024, kernel_size=1)(out)
        out = (F.relu(out))
        out = nn.BatchNorm1d(1024)(out)

        out = nn.MaxPool1d(kernel_size=5076)(out).view(2, 1024)

        out = nn.Linear(1024, 512)(out)
        out = (F.relu(out))
        out = nn.BatchNorm1d(512)(out)

        out = nn.Linear(512, 256)(out)
        out = (F.relu(out))
        out = nn.BatchNorm1d(256)(out)

        out = nn.Linear(256, 9)(out)

        iden = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
        print(f"{iden} <- identity matrix")
        out = out.view(-1, 3, 3)  + iden

        print("--------------------------- bmm ------------------------------")

        x = torch.bmm(x.transpose(2, 1), out).transpose(2, 1)

        x = F.relu(nn.Conv1d(3, 64, kernel_size=1)(x))
        x = nn.BatchNorm1d(64)(x)

        x = F.relu(nn.Conv1d(64, 64, kernel_size=1)(x))
        x = nn.BatchNorm1d(64)(x)

        print("--------------------------- Tnet 2 ------------------------------")

        bs = x.shape[0]
        out = nn.Conv1d(64, 64, kernel_size=1)(x)
        out = (F.relu(out))
        out = nn.BatchNorm1d(64)(out)

        out = nn.Conv1d(64, 128, kernel_size=1)(out)
        out = (F.relu(out))
        out = nn.BatchNorm1d(128)(out)

        out = nn.Conv1d(128, 1024, kernel_size=1)(out)
        out = (F.relu(out))
        out = nn.BatchNorm1d(1024)(out)

        out = nn.MaxPool1d(kernel_size=5076)(out).view(2, 1024)

        out = nn.Linear(1024, 512)(out)
        out = (F.relu(out))
        out = nn.BatchNorm1d(512)(out)

        out = nn.Linear(512, 256)(out)
        out = (F.relu(out))
        out = nn.BatchNorm1d(256)(out)

        out = nn.Linear(256, 4096)(out)

        iden = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)
        print(f"{iden} <- identity matrix")
        out = out.view(-1, 64, 64)  + iden

        print("--------------------------- bmm ------------------------------")
        
        x = torch.bmm(x.transpose(2, 1), out).transpose(2, 1)
        print(x[1][1])
        print(x.shape)
        local_features = x.clone()

        x = F.relu(nn.Conv1d(64, 64, kernel_size=1)(x))
        x = nn.BatchNorm1d(64)(x)
        x = F.relu(nn.Conv1d(64, 128, kernel_size=1)(x))
        x = nn.BatchNorm1d(128)(x)
        x = F.relu(nn.Conv1d(128, 1024, kernel_size=1)(x))
        x = nn.BatchNorm1d(1024)(x)

        global_features, critical_indexes = nn.MaxPool1d(kernel_size=5076, return_indices=True)(x)    #get global feature vector and critical indexes
        global_features = global_features.view(bs, -1)
        critical_indexes = critical_indexes.view(bs, -1)

        features = torch.cat((local_features, global_features.unsqueeze(-1).repeat(1, 1, 5076)), dim=1)
        print(features.shape)

        print("-----------------------------detecthead-----------------------")
        print(f"data shape -> {data.shape}")
        print(f"global_features-> {global_features.shape}")
        print(f"global_features len-> {len(global_features)}")
        loc_layers = nn.ModuleList([nn.Conv1d(len(global_features), 6 * 3, 1)])
        
        loc_preds = []
        for l in loc_layers:
            loc_pred = l(global_features)
            loc_preds.append(loc_pred.reshape(-1, 6))
        loc_preds = torch.cat(loc_preds, dim=0)
        
        print("   ")
        print("-----------------------------location predictions-----------------------")
        print("   ")
        print(loc_pred)
        print(loc_pred.shape)
        print(len(loc_preds))
        print(loc_preds.shape)
        print(loc_preds[1])

        print("   ")
        print("----------------------------- anchor box match-----------------------")
        print("   ")

        anchors = AnchorBox(infos)()
        matched_gt_boxes = []
        a=0
        print(f" y = {y}")
        print(f"anchor[0] = {anchors[0]}")
        iou_scores = []
        for i, anchor in enumerate(anchors):
            a=1
           # anchor = [0.20, 0.20, -2.20, 13, 13, 13]
            for gt_box in y:
                if not np.all(np.isnan(np.array(gt_box))): 
                    iou_scores = [compute_iou(anchor, box) for box in gt_box]

            if len(iou_scores) != 0:
                tensor_stack = torch.stack(iou_scores)
                non_nan_tensors = tensor_stack[~torch.isnan(tensor_stack)]
                max_iou = torch.max(non_nan_tensors)
                #print(f"iou_scores - {iou_scores}")
                if max_iou >= 0.05:
                    matched_gt_boxes.append(gt_box[iou_scores.index(max_iou)])
            print(f"\r{i}-th anchor loaded", end=" ", flush=True)
            
        centers['x'], centers['y'], centers['z'] = anchors[:, 0], anchors[:, 1], anchors[:, 2]
        
        df_centers = pd.DataFrame(centers)
        sns.scatterplot(x='z', y='y', data=df_centers)
        # print(f"Matched boxes length -> {len(iou_scores)}")
        # print(f"IoU scores first value -> {iou_scores[0]}")
        print(f"Matched gt_boxes length = {len(matched_gt_boxes)} ")
        plt.show()
        cnt += 1
        if(cnt >= 1):
            break