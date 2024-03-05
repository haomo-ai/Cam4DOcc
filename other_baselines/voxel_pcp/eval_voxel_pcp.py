# Developed by Junyi Ma
# Cam4DOcc: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications
# https://github.com/haomo-ai/Cam4DOcc

import numpy as np
import os
import copy
from tqdm import trange
import open3d as o3d
from nuscenes import NuScenes
from pyquaternion import Quaternion

# Setups =================================================================================================
test_idx_dir = "../../data/cam4docc/test_ids/"
test_results_dir = "../../data/cam4docc/pcpnet_results/"
occ_path = "../../data/nuScenes-Occupancy"

test_seqs = os.listdir(test_idx_dir)
test_segmentations = os.listdir(test_results_dir)
pc_range= np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
dimension = [512, 512, 40]
grid_size= np.array(dimension)
voxel_size = (pc_range[3:] -pc_range[:3]) / grid_size
future_ious = [0, 0, 0, 0]

nusc = NuScenes(version='v1.0-trainval', dataroot="../../data/nuscenes", verbose=False)
# ========================================================================================================

lidar_token2sample_token = {}
for i in range(len(nusc.sample)):  
    my_sample = nusc.sample[i]
    frame_token = my_sample['token']
    lidar_token = my_sample['data']['LIDAR_TOP']
    lidar_token2sample_token[lidar_token] = frame_token

def voxel2world(voxel):
    """
    voxel: [N, 3]
    """
    return voxel *voxel_size[None, :] + pc_range[:3][None, :]

def world2voxel(world):
    """
    world: [N, 3]
    """
    return (world - pc_range[:3][None, :]) / voxel_size[None, :]

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

def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    
    return processed_label

def get_ego2lidar_pose(rec):
    lidar_top_data = nusc.get('sample_data', rec['data']['LIDAR_TOP'])
    lidar2ego_translation = nusc.get('calibrated_sensor', lidar_top_data['calibrated_sensor_token'])['translation']
    lidar2ego_rotation =  nusc.get('calibrated_sensor', lidar_top_data['calibrated_sensor_token'])['rotation']
    trans = -np.array(lidar2ego_translation)
    rot = Quaternion(lidar2ego_rotation).inverse
    return trans, rot

def get_lidar_pose(rec):
    current_sample = nusc.get('sample', rec['token'])
    egopose = nusc.get('ego_pose', nusc.get('sample_data', current_sample['data']['LIDAR_TOP'])['ego_pose_token'])
    trans = -np.array(egopose['translation'])
    rot = Quaternion(egopose['rotation']).inverse  
    return trans, rot

for i in trange(len(test_seqs)):
    test_seqs_idxs = np.load(os.path.join(test_idx_dir, test_seqs[i]))['arr_0']
    scene_token_present = test_seqs[i].split("_")[0]
    lidar_token_present = test_seqs[i].split("_")[1][:-4]

    # transform past point clouds to the present frame
    # point cloud prediction baseline is limited by sparsity of laser points, so we aggregate
    # past point clouds to mitigate in this version
    # More reasonable versions will be released
    past_voxels = []
    for t in range(1, 4):
        scene_token_ = test_seqs_idxs[t-1].split("_")[0]
        lidar_token_ = test_seqs_idxs[t-1].split("_")[1]
        point_file = test_results_dir+"point_clouds/"+scene_token_present+"_"+lidar_token_present+"/past/00000"+str(t)+".ply"
        label_file = test_results_dir+"saved_labels/"+scene_token_present+"_"+lidar_token_present+"/past/00000"+str(t)+".label"
        
        pcd_load = o3d.io.read_point_cloud(point_file)
        xyz_load = np.asarray(pcd_load.points)

        sample_token_present = lidar_token2sample_token[lidar_token_present]
        rec_present = nusc.get('sample', sample_token_present)
        translation_present, rotation_present = get_lidar_pose(rec_present)
        ego2lidar_translation_present, ego2lidar_rotation_present = get_ego2lidar_pose(rec_present)

        sample_token_ = lidar_token2sample_token[lidar_token_]
        rec_ = nusc.get('sample', sample_token_)
        translation_, rotation_ = get_lidar_pose(rec_)
        ego2lidar_translation_, ego2lidar_rotation_ = get_ego2lidar_pose(rec_)

        present_global2ego = [translation_present, rotation_present]
        present_ego2lidar = [ego2lidar_translation_present, ego2lidar_rotation_present]
        cur_global2ego = [translation_, rotation_]
        cur_ego2lidar = [ego2lidar_translation_, ego2lidar_rotation_]
        pcd_np_cor = np.dot(cur_ego2lidar[1].inverse.rotation_matrix, xyz_load.T)
        pcd_np_cor = pcd_np_cor.T
        pcd_np_cor = pcd_np_cor - cur_ego2lidar[0]
        pcd_np_cor = np.dot(cur_global2ego[1].inverse.rotation_matrix, pcd_np_cor.T)
        pcd_np_cor = pcd_np_cor.T
        pcd_np_cor = pcd_np_cor - cur_global2ego[0]
        pcd_np_cor = pcd_np_cor + present_global2ego[0]
        pcd_np_cor = np.dot(present_global2ego[1].rotation_matrix, pcd_np_cor.T)
        pcd_np_cor = pcd_np_cor.T
        pcd_np_cor = pcd_np_cor + present_ego2lidar[0]   # trans
        pcd_np_cor = np.dot(present_ego2lidar[1].rotation_matrix, pcd_np_cor.T)
        xyz_load = pcd_np_cor.T   

        xyz_load = world2voxel(xyz_load)
        label = np.fromfile(label_file, dtype=np.uint32)
        label = label.reshape((-1,1))
        segmentation_t = np.concatenate((xyz_load, label), axis=-1)
        kept = (segmentation_t[:,0]>0) & (segmentation_t[:,0]<dimension[0]) & (segmentation_t[:,1]>0) & (segmentation_t[:,1]<dimension[1])  & (segmentation_t[:,2]>0) & (segmentation_t[:,2]<dimension[2])
        segmentation_t = segmentation_t[kept]
        past_voxels.append(segmentation_t)

    past_voxel_aggregated = np.concatenate(past_voxels, axis=0)

    # for future forecasting
    for t in range(3, 7):
        scene_token_ = test_seqs_idxs[t].split("_")[0]
        lidar_token_ = test_seqs_idxs[t].split("_")[1]

        point_file = test_results_dir+"point_clouds/"+scene_token_present+"_"+lidar_token_present+"/pred/00000"+str(t-3)+".ply"
        label_file = test_results_dir+"saved_labels/"+scene_token_present+"_"+lidar_token_present+"/pred/00000"+str(t-3)+".label"

        pcd_load = o3d.io.read_point_cloud(point_file)
        xyz_load = np.asarray(pcd_load.points)
        xyz_load = world2voxel(xyz_load)

        label = np.fromfile(label_file, dtype=np.uint32)
        label = label.reshape((-1,1))

        segmentation_t = np.concatenate((xyz_load, label), axis=-1)
        kept = (segmentation_t[:,0]>0) & (segmentation_t[:,0]<dimension[0]) & (segmentation_t[:,1]>0) & (segmentation_t[:,1]<dimension[1])  & (segmentation_t[:,2]>0) & (segmentation_t[:,2]<dimension[2])
        segmentation_t = segmentation_t[kept]
        segmentation_t = np.concatenate((segmentation_t, past_voxel_aggregated), axis=0)

        pred_segmentation = np.zeros((dimension[0], dimension[1], dimension[2]))
        pred_segmentation[segmentation_t[:, 0].astype(int), segmentation_t[:, 1].astype(int), segmentation_t[:, 2].astype(int)] = segmentation_t[:, -1]

        # eval according to setups
        # hardcoding for classes of interest
        for otheridx in [0,1,8,11,12,13,14,15,16,17,18,255]:
            pred_segmentation[pred_segmentation==otheridx] = 0
        for vehidx in [2,3,4,5,6,7,9,10]:
            pred_segmentation[pred_segmentation==vehidx] = 1

        # load nuScenes-Occupancy
        rel_path = 'scene_{0}/occupancy/{1}.npy'.format(scene_token_, lidar_token_)
        gt_segmentation_file = os.path.join(occ_path, rel_path)
        pcd = np.load(gt_segmentation_file)

        pcd_label = pcd[..., -1:]
        pcd_label[pcd_label==0] = 255
        pcd_np_cor = voxel2world(pcd[..., [2,1,0]] + 0.5)
        pcd_np_cor = world2voxel(pcd_np_cor)

        # make sure the point is in the grid
        pcd_np_cor = np.clip(pcd_np_cor, np.array([0,0,0]), grid_size - 1)
        transformed_occ = copy.deepcopy(pcd_np_cor)
        pcd_np = np.concatenate([pcd_np_cor, pcd_label], axis=-1)

        # 255: noise, 1-16 normal classes, 0 unoccupied
        pcd_np = pcd_np[np.lexsort((pcd_np_cor[:, 0], pcd_np_cor[:, 1], pcd_np_cor[:, 2])), :]
        pcd_np = pcd_np.astype(np.int64)
        processed_label = np.ones(grid_size, dtype=np.uint8) * 0.0
        processed_label = nb_process_label(processed_label, pcd_np)
        # Opt.
        # processed_label[pcd_np[:, 0].astype(int), pcd_np[:, 1].astype(int), pcd_np[:, 2].astype(int)] = pcd_np[:, -1]

        for otheridx in [0,1,8,11,12,13,14,15,16,17,18,255]:
            processed_label[processed_label==otheridx] = 0
        for vehidx in [2,3,4,5,6,7,9,10]:
            processed_label[processed_label==vehidx] = 1

        hist_cur, iou_per_pred = fast_hist(pred_segmentation.astype(int), processed_label.astype(int), max_label=2)

        if t <= 3:
            future_ious[0] = future_ious[0] + hist_cur
        if t <= 4:
            future_ious[1] = future_ious[1] + hist_cur
        if t <= 5:
            future_ious[2] = future_ious[2] + hist_cur
        if t <= 6:
            future_ious[3] = future_ious[3] + hist_cur
            
for t in range(len(future_ious)):
    print("ious for step "+str(t), cm_to_ious(future_ious[t]))