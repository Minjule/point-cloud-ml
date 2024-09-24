import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from anchors import AnchorBox

infos = {
    'grid_size': (10, 10, 10),  # Number of grid points in (x, y, z) dimensions
    'min_sizes': [0.13, 0.13, 0.13],  # Min sizes for anchor boxes in 3D
    'max_sizes': [0.18, 0.18, 0.18],  # Max sizes for anchor boxes in 3D
    'aspect_ratios': [(1, 1, 1), (1.5, 1, 1), (1, 1.5, 1)],  # Aspect ratios for anchor boxes
    'steps': [0.5, 0.5, 0.5],  # Step size for the anchor grid in 3D space
    'point_cloud_range': [-8.90265846,  -8.36585426, -10.08037663, 10.86396313,  5.29646969, 14.27643776],  # 3D space range in (x_min, y_min, z_min, x_max, y_max, z_max)
    'clip': True  
}

class Tnet(nn.Module):
    def __init__(self, dim):

        super(Tnet, self).__init__()

        self.dim = dim           #dimensions for transform matrix

        self.conv1 = nn.Conv1d(dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, dim**2)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.max_pool = nn.MaxPool1d(kernel_size=5076)
        

    def forward(self, x):

        #shared MLP layers (conv1d)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))

        #max pool
        x = self.max_pool(x).view(4, 1024)
        
        #MLP
        x = self.bn4(F.relu(self.linear1(x)))
        x = self.bn5(F.relu(self.linear2(x)))
        x = self.linear3(x)

        # initialize identity matrix
        iden = torch.eye(self.dim, requires_grad=True).repeat(4, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()

        x = x.view(-1, self.dim, self.dim) + iden

        return x


class PointNetBackbone(nn.Module):
    
    def __init__(self, num_points=5076, num_global_feats=1024, local_feat=False):

        super(PointNetBackbone, self).__init__()

        self.num_points = num_points                      #if true concat local and global features
        self.num_global_feats = num_global_feats
        self.local_feat = local_feat

        self.tnet1 = Tnet(dim=3)   #Spatial Transformer Networks (T-nets)
        self.tnet2 = Tnet(dim=64)

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)      #shared MLP 1
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)


        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)     #shared MLP 2
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv1d(128, self.num_global_feats, kernel_size=1)
        

        self.bn1 = nn.BatchNorm1d(64)                     #batch norms for both shared MLPs
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(self.num_global_feats)

        # max pool to get the global features
        self.max_pool = nn.MaxPool1d(kernel_size=num_points, return_indices=True)

    
    def forward(self, x):
        bs = x.shape[0]

        A_input = self.tnet1(x)
        x = torch.bmm(x.transpose(2, 1), A_input).transpose(2, 1)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        
        A_feat = self.tnet2(x)
        x = torch.bmm(x.transpose(2, 1), A_feat).transpose(2, 1)

        local_features = x.clone()     #store local point features for segmentation head

        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))

        global_features, critical_indexes = self.max_pool(x)    #get global feature vector and critical indexes
        global_features = global_features.view(bs, -1)
        critical_indexes = critical_indexes.view(bs, -1)

        if self.local_feat:
            features = torch.cat((local_features, global_features.unsqueeze(-1).repeat(1, 1, self.num_points)), dim=1)
            return features, critical_indexes, A_feat

        else:
            return global_features, critical_indexes, A_feat

class PointNetDetectHead(nn.Module):
    def __init__(self, num_points=5076, num_global_feats=1024, num_defaults=3):
        super(PointNetDetectHead, self).__init__()

        self.backbone = PointNetBackbone(num_points = num_points, num_global_feats =num_global_feats, local_feat=False)
        self.loc_layers = nn.ModuleList([nn.Conv1d(4, 6 * num_defaults, 1)])
    
    def generate_anchors(self):
        boxes = AnchorBox(infos)()
        return boxes
    
    def forward(self, x):
        loc_preds = []
        out = self.backbone(x)
        for l in self.loc_layers:
            print(out[0].shape)
            loc_pred = l(out[0])
            loc_preds.append(loc_pred.reshape(-1, 6))
        loc_preds = torch.cat(loc_preds, dim=0)
        return loc_preds
    
    def compute_intersection_volume(self, box1, box2):
        # Unpack box 1 and box 2
        center_x1, center_y1, center_z1, w1, h1, d1 = box1
        center_x2, center_y2, center_z2, w2, h2, d2 = box2
        
        # Calculate min/max boundaries for each box
        x1_min, x1_max = center_x1 - w1 / 2, center_x1 + w1 / 2
        y1_min, y1_max = center_y1 - h1 / 2, center_y1 + h1 / 2
        z1_min, z1_max = center_z1 - d1 / 2, center_z1 + d1 / 2

        x2_min, x2_max = center_x2 - w2 / 2, center_x2 + w2 / 2
        y2_min, y2_max = center_y2 - h2 / 2, center_y2 + h2 / 2
        z2_min, z2_max = center_z2 - d2 / 2, center_z2 + d2 / 2
        
        # Calculate the overlap in each dimension
        x_overlap = torch.max(torch.zeros_like(x1_max), torch.min(x1_max, x2_max) - torch.max(x1_min, x2_min))
        y_overlap = torch.max(torch.zeros_like(y1_max), torch.min(y1_max, y2_max) - torch.max(y1_min, y2_min))
        z_overlap = torch.max(torch.zeros_like(z1_max), torch.min(z1_max, z2_max) - torch.max(z1_min, z2_min))
        
        # Compute intersection volume
        intersection_volume = x_overlap * y_overlap * z_overlap
        
        return intersection_volume
    
    def compute_iou(self, box1, box2):
        # This is a simplified IoU function. You need to define the exact intersection
        # volume calculation for 3D bounding boxes depending on your coordinate format.
        intersection_vol = self.compute_intersection_volume(box1, box2)  # You need this function
        vol_box1 = box1[3] * box1[4] * box1[5]  # width * height * depth
        vol_box2 = box2[3] * box2[4] * box2[5]
        union_vol = vol_box1 + vol_box2 - intersection_vol
        return intersection_vol / union_vol

    def match_anchors_to_ground_truth(self, anchors, loc_preds, gt_boxes, iou_threshold=0.5):
        # anchors: predefined anchor boxes (3D anchors)
        # loc_preds: predicted bounding boxes from the model
        # gt_boxes: ground truth boxes
        matched_gt_boxes = []
        for i, anchor in enumerate(anchors):
            iou_scores = [self.compute_iou(loc_preds[i], gt_box) for gt_box in gt_boxes]
            max_iou = torch.max(iou_scores)
            if max_iou >= iou_threshold:
                matched_gt_boxes.append(gt_boxes[iou_scores.index(max_iou)])
            else:
                matched_gt_boxes.append(None)  # Negative match, no object
        return matched_gt_boxes

    def ssd_loss(self, loc_preds, matched_boxes):

        pos_mask = matched_boxes != None  # Positive samples mask
        
        # 1. Localization loss (Smooth L1 loss between matched GT boxes and predictions)
        loc_loss = F.smooth_l1_loss(loc_preds[pos_mask], matched_boxes[pos_mask], reduction='sum')
        
        return loc_loss

    def non_maximum_suppression(self, pred_boxes, cls_scores, iou_threshold=0.5):
        indices = cls_scores.argsort(descending=True)  # Sort in descending order
        keep_boxes = []
        while len(indices) > 0:
            current_idx = indices[0]
            keep_boxes.append(pred_boxes[current_idx])
            remaining_boxes = pred_boxes[indices[1:]]
            ious = [self.compute_iou(pred_boxes[current_idx], box) for box in remaining_boxes]
            indices = indices[1:][[iou < iou_threshold for iou in ious]]
        return keep_boxes
        
        
class PointNetClassHead(nn.Module):

    def __init__(self, num_points=2500, num_global_feats=1024, k=2):
        super(PointNetClassHead, self).__init__()

        self.backbone = PointNetBackbone(num_points, num_global_feats, local_feat=False)

        #MLP
        self.linear1 = nn.Linear(num_global_feats, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, k)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(p=0.3)
        

    def forward(self, x):
        x, crit_idxs, A_feat = self.backbone(x) 

        x = self.bn1(F.relu(self.linear1(x)))
        x = self.bn2(F.relu(self.linear2(x)))
        x = self.dropout(x)
        x = self.linear3(x)

        #return logits
        return x, crit_idxs, A_feat



class PointNetSegHead(nn.Module):
    def __init__(self, num_points=2500, num_global_feats=1024, m=2):
        super(PointNetSegHead, self).__init__()

        self.num_points = num_points
        self.m = m

        self.backbone = PointNetBackbone(num_points, num_global_feats, local_feat=True)

        #MLP
        num_features = num_global_feats + 64 # local and global features
        self.conv1 = nn.Conv1d(num_features, 512, kernel_size=1)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=1)
        self.conv4 = nn.Conv1d(128, m, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)


    def forward(self, x):
        
        # get combined features
        x, crit_idxs, A_feat = self.backbone(x) 

        #MLP
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.conv4(x)

        x = x.transpose(2, 1)
        
        return x, crit_idxs, A_feat

class PointNet(nn.Module):
    def __init__(self, dim, num_points=2500, num_global_feats=1024, local_feat=False):
        super(PointNet, self).__init__()

        self.detect = PointNetDetectHead(num_points, num_global_feats, local_feat=local_feat, num_defaults=3)

    def forward(self, x):
        preds = self.detect(x)
        return preds

     
def main():
    test_data = torch.rand(32, 3, 2500)

    ## test T-net
    tnet = Tnet(dim=3)
    transform = tnet(test_data)
    print(f'T-net output shape: {transform.shape}')

    ## test backbone
    pointfeat = PointNetBackbone(local_feat=False)
    out, _, _ = pointfeat(test_data)
    print(f'Global Features shape: {out.shape}')

    pointfeat = PointNetBackbone(local_feat=True)
    out, _, _ = pointfeat(test_data)
    print(f'Combined Features shape: {out.shape}')

    # test on single batch (should throw error if there is an issue)
    pointfeat = PointNetBackbone(local_feat=True).eval()
    out, _, _ = pointfeat(test_data[0, :, :].unsqueeze(0))

    ## test classification head
    classifier = PointNetClassHead(k=5)
    out, _, _ = classifier(test_data)
    print(f'Class output shape: {out.shape}')

    classifier = PointNetClassHead(k=5).eval()
    out, _, _ = classifier(test_data[0, :, :].unsqueeze(0))

    ## test segmentation head
    seg = PointNetSegHead(m=3)
    out, _, _ = seg(test_data)
    print(f'Seg shape: {out.shape}')


if __name__ == '__main__':
    gpus = [0]
    device = torch.device(f'cuda:{gpus[0]}' if torch.cuda.is_available() else 'cpu')
    detect = PointNetBackbone(local_feat=True).to(device=device)
    print(detect)

