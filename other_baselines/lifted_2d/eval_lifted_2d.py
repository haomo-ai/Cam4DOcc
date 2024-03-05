# Developed by Junyi Ma
# Cam4DOcc: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications
# https://github.com/haomo-ai/Cam4DOcc

from tqdm import trange
import numpy as np
from nuscenes import NuScenes
import os
import torch
import torch.nn.functional as F
import copy
from pyquaternion import Quaternion

# Setups =================================================================================================
test_idx_dir = "../../data/cam4docc/test_ids/"
test_results_dir = "../../data/cam4docc/powerbev_results/"
gt_dir = "../../data/cam4docc/MMO/segmentation/"

test_seqs = os.listdir(test_idx_dir)
test_segmentations = os.listdir(test_results_dir)
dimension = [512, 512, 40]
future_ious = [0, 0, 0, 0]

voxel_size = np.array([0.2,0.2,0.2])
pc_range = np.array([-50, -50, 0, 50, 50, 0])
voxel_size_new = np.array([0.2,0.2,0.2])
pc_range_new = np.array([-51.2, -51.2, -5, 51.2, 51.2, 3])

# 10*0.2=2m 
# You can modify the parameters to show the changes with variable heights for lifting
hmin = -1
hmax = 9

nusc = NuScenes(version='v1.0-trainval', dataroot="../../data/nuscenes", verbose=False)
# ========================================================================================================

def cm_to_ious(cm):
    mean_ious = []
    cls_num = len(cm)
    for i in range(cls_num):
        tp = cm[i, i]
        p = cm[:, i].sum()
        g = cm[i, :].sum()
        union = p + g - tp
        mean_ious.append(tp / union)
    return mean_ious

def fast_hist(pred, label, max_label=18): 
    pred = copy.deepcopy(pred.flatten())
    label = copy.deepcopy(label.flatten())
    bin_count = np.bincount(max_label * label.astype(int) + pred, minlength=max_label ** 2)
    iou_per_pred = (bin_count[-1]/(bin_count[-1]+bin_count[1]+bin_count[2]))
    return bin_count[:max_label ** 2].reshape(max_label, max_label),iou_per_pred

for i in trange(len(test_seqs)):
    segmentation_file = test_results_dir + test_seqs[i]
    instance_seq = np.load(segmentation_file)['arr_0']
    instance_seq = torch.from_numpy(instance_seq)

    test_seqs_idxs = np.load(test_idx_dir+test_seqs[i])["arr_0"]
    gt_segmentation_file = os.path.join(gt_dir, test_seqs[i])
    gt_segmentation_seqs = np.load(gt_segmentation_file, allow_pickle=True)['arr_0']

    for t in range(3, 7):
        scene_token_cur = test_seqs_idxs[t].split("_")[0]
        lidar_token_cur = test_seqs_idxs[t].split("_")[1]

        instance_ = instance_seq[0,(t-1)].unsqueeze(0)  # t-1 -> t
        instance_ = instance_.unsqueeze(0)
        instance_ = F.interpolate(instance_.float(), size=[500, 500], mode='nearest').contiguous() # Note: default PowerBEV has different ranges with OCFNet
        instance_ = instance_.squeeze(0)

        x_grid = torch.linspace(0, 500-1, 500, dtype=torch.float)
        x_grid = x_grid.view(500, 1).expand(500,500)
        y_grid = torch.linspace(0, 500-1,500, dtype=torch.float)
        y_grid = y_grid.view(1, 500).expand(500,500)
        mesh_grid_2d = torch.stack((x_grid, y_grid), -1)
        mesh_grid_2d = mesh_grid_2d.view(-1, 2)
        instance_ = instance_.view(-1, 1)

        semantics_lifted = []
        for ii in range(hmin, hmax): 
            semantics_lifted_ = torch.cat((mesh_grid_2d, ii*torch.ones_like(mesh_grid_2d[:,0:1])),dim=-1)
            semantics_lifted_ = torch.cat((semantics_lifted_, instance_),dim=-1)
            semantics_lifted.append(semantics_lifted_)

        semantics_lifted = np.array(torch.cat(semantics_lifted, dim=0))

        kept = semantics_lifted[:,-1]!=0
        semantics_lifted = semantics_lifted[kept]
        if semantics_lifted.shape[0] == 0:
            semantics_lifted = np.zeros((1,4))

        lidar_sample = nusc.get('sample_data', lidar_token_cur)
        lidar_sample_calib = nusc.get('calibrated_sensor', lidar_sample['calibrated_sensor_token'])
        lidar_sensor_rotation = Quaternion(lidar_sample_calib['rotation'])
        lidar_sensor_translation = np.array(lidar_sample_calib['translation'])[:, None]
        lidar_to_lidarego = np.vstack([
            np.hstack((lidar_sensor_rotation.rotation_matrix, lidar_sensor_translation)),
            np.array([0, 0, 0, 1])
        ])
        lidarego_to_lidar = np.linalg.inv(lidar_to_lidarego)
        points = np.ones_like(semantics_lifted)
        points[:,:3] = semantics_lifted[:,:3]
        points[:,:3] = points[:,:3] * voxel_size[None, :] + pc_range[:3][None, :] 
        points = lidarego_to_lidar @ points.T
        semantics_lifted_transformed = np.ones_like(semantics_lifted)
        semantics_lifted_transformed[:,:3] = (points.T)[:,:3]
        semantics_lifted_transformed[:,-1] = semantics_lifted[:,-1]
        semantics_lifted_transformed[:,:3] = (semantics_lifted_transformed[:,:3] - pc_range_new[:3][None, :]) / voxel_size_new[None, :] 

        pred_segmentation = np.zeros((dimension[0], dimension[1], dimension[2]))
        for j in range(semantics_lifted_transformed.shape[0]):
            cur_ind = semantics_lifted_transformed[j, :3].astype(int)
            cur_label = semantics_lifted_transformed[j, -1]
            if cur_label != 0:
                pred_segmentation[cur_ind[0],cur_ind[1],cur_ind[2]] = 1

        gt_segmentation = np.zeros((dimension[0], dimension[1], dimension[2]))
        gt_segmentation_raw = gt_segmentation_seqs[t].cpu().numpy()
        gt_segmentation[gt_segmentation_raw[:,0].astype(int),gt_segmentation_raw[:,1].astype(int),gt_segmentation_raw[:,2].astype(int)] = gt_segmentation_raw[:, -1]

        hist_cur, iou_per_pred = fast_hist(pred_segmentation.astype(int), gt_segmentation.astype(int), max_label=2)
        
        if t <= 3:
            future_ious[0] = future_ious[0] + hist_cur
        if t <= 4:
            future_ious[1] = future_ious[1] + hist_cur
        if t <= 5:
            future_ious[2] = future_ious[2] + hist_cur
        if t <= 6:
            future_ious[3] = future_ious[3] + hist_cur

for t in range(len(future_ious)):
    print("iou for step "+str(t), cm_to_ious(future_ious[t]))