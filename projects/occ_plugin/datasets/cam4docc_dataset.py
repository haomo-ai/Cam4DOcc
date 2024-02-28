# Developed by Junyi Ma based on the codebase of OpenOccupancy and PowerBEV
# Cam4DOcc: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications
# https://github.com/haomo-ai/Cam4DOcc

import numpy as np
from mmcv.runner import get_dist_info
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import os
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from projects.occ_plugin.utils.formating import cm_to_ious, format_iou_results
from projects.occ_plugin.utils.geometry import convert_egopose_to_matrix_numpy, invert_matrix_egopose_numpy
from nuscenes import NuScenes
from pyquaternion import Quaternion
import torch
import random
import time

@DATASETS.register_module()
class Cam4DOccDataset(NuScenesDataset):
    def __init__(self, occ_size, pc_range, occ_root, idx_root, time_receptive_field, n_future_frames, classes, use_separate_classes,
                  train_capacity, test_capacity , **kwargs):
        
        '''
        Cam4DOccDataset contains sequential occupancy states as well as instance flow for training occupancy forecasting models. We unify the related operations in the LiDAR coordinate system following OpenOccupancy.

        occ_size: number of grids along H W L, default: [512, 512, 40]
        pc_range: predefined ranges along H W L, default: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        occ_root: data path of nuScenes-Occupancy
        idx_root: save path of test indexes
        time_receptive_field: number of historical frames used for forecasting (including the present one), default: 3
        n_future_frames: number of forecasted future frames, default: 4
        classes: predefiend categories in GMO
        use_separate_classes: separate movable objects instead of the general one
        train_capacity: number of sequences used for training, default: 23930
        test_capacity: number of sequences used for testing, default: 5119
        '''
        
        self.train_capacity = train_capacity
        self.test_capacity = test_capacity

        super().__init__(**kwargs)

        rank, world_size = get_dist_info()

        self.time_receptive_field = time_receptive_field
        self.n_future_frames = n_future_frames
        self.sequence_length = time_receptive_field + n_future_frames

        if rank == 0:
            print("-------------")
            print("use past " + str(self.time_receptive_field) + " frames to forecast future " + str(self.n_future_frames) + " frames")
            print("-------------")

        self.data_infos = list(sorted(self.data_infos, key=lambda e: e['timestamp']))
        self.data_infos = self.data_infos[::self.load_interval]
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.occ_root = occ_root
        self.idx_root = idx_root
        self.classes = classes
        self.use_separate_classes = use_separate_classes

        self.indices = self.get_indices()
        self.present_scene_lidar_token = " "
        self._set_group_flag()

        # load origin nusc dataset for instance annotation
        self.nusc = NuScenes(version='v1.0-trainval', dataroot='/mnt/share_disk/dataset/nuscenes/data', verbose=False)
        if self.test_mode:
            self.chosen_list = random.sample(range(0, self.test_capacity) , self.test_capacity)
            self.chosen_list_num = len(self.chosen_list)
        else:
            self.chosen_list = random.sample(range(0, self.train_capacity) , self.train_capacity)
            self.chosen_list_num = len(self.chosen_list)
    
    def _set_group_flag(self):
        if self.test_mode:
            self.flag = np.zeros(self.test_capacity, dtype=np.uint8)
        else:
            self.flag = np.zeros(self.train_capacity, dtype=np.uint8)

    def __len__(self):
        if self.test_mode:
            return self.test_capacity
        else:
            return self.train_capacity

    def __getitem__(self, idx):
     
        idx = int(self.chosen_list[idx])

        self.egopose_list = []
        self.ego2lidar_list = []
        self.visible_instance_set = set()
        self.instance_dict = {}

        if self.test_mode:
            return self.prepare_test_data(idx)
            
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                idx = int(self.chosen_list[idx])
                continue
            
            return data

    def get_indices(self):
        '''
        Generate sequential indexes for training and testing
        '''
        indices = []
        for index in range(len(self.data_infos)): 
            is_valid_data = True
            previous_rec = None
            current_indices = []
            for t in range(self.sequence_length):
                index_t = index + t
                # Going over the dataset size limit.
                if index_t >= len(self.data_infos):
                    is_valid_data = False
                    break
                rec = self.data_infos[index_t]
                # Check if scene is the same
                if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
                    is_valid_data = False
                    break

                current_indices.append(index_t)
                previous_rec = rec

            if is_valid_data:
                indices.append(current_indices)

        return np.asarray(indices)

    def get_lidar_pose(self, rec):
        '''
        Get global poses for following bbox transforming
        '''
        ego2global_translation = rec['ego2global_translation']
        ego2global_rotation = rec['ego2global_rotation']
        trans = -np.array(ego2global_translation)
        rot = Quaternion(ego2global_rotation).inverse
        
        return trans, rot
    
    def get_ego2lidar_pose(self, rec):
        '''
        Get LiDAR poses in ego system
        '''
        lidar2ego_translation = rec['lidar2ego_translation']
        lidar2ego_rotation = rec['lidar2ego_rotation']
        trans = -np.array(lidar2ego_translation)
        rot = Quaternion(lidar2ego_rotation).inverse
        return trans, rot

    def record_instance(self, idx, instance_map):
        """
        Record information about each visible instance in the sequence and assign a unique ID to it
        """
        rec = self.data_infos[idx]
        translation, rotation = self.get_lidar_pose(rec)
        self.egopose_list.append([translation, rotation])
        ego2lidar_translation, ego2lidar_rotation = self.get_ego2lidar_pose(rec)
        self.ego2lidar_list.append([ego2lidar_translation, ego2lidar_rotation])

        current_sample = self.nusc.get('sample', rec['token'])
        for annotation_token in current_sample['anns']:
            annotation = self.nusc.get('sample_annotation', annotation_token)
            # Instance extraction for Cam4DOcc-V1 
            # Filter out all non vehicle instances
            # if 'vehicle' not in annotation['category_name']:
            #     continue
            gmo_flag = False
            for class_name in self.classes:
                if class_name in annotation['category_name']:
                    gmo_flag = True
                    break
            if not gmo_flag:
                continue
            # Specify semantic id if use_separate_classes
            semantic_id = 1
            if self.use_separate_classes:
                if 'bicycle' in annotation['category_name']:
                    semantic_id = 1
                elif 'bus'  in annotation['category_name']:
                    semantic_id = 2
                elif 'car'  in annotation['category_name']:
                    semantic_id = 3
                elif 'construction'  in annotation['category_name']:
                    semantic_id = 4
                elif 'motorcycle'  in annotation['category_name']:
                    semantic_id = 5
                elif 'trailer'  in annotation['category_name']:
                    semantic_id = 6
                elif 'truck'  in annotation['category_name']:
                    semantic_id = 7
                elif 'pedestrian'  in annotation['category_name']:
                    semantic_id = 8

            # Filter out invisible vehicles
            FILTER_INVISIBLE_VEHICLES = True
            if FILTER_INVISIBLE_VEHICLES and int(annotation['visibility_token']) == 1 and annotation['instance_token'] not in self.visible_instance_set:
                continue
            # Filter out vehicles that have not been seen in the past
            if self.counter >= self.time_receptive_field and annotation['instance_token'] not in self.visible_instance_set:
                continue
            self.visible_instance_set.add(annotation['instance_token'])

            if annotation['instance_token'] not in instance_map:
                instance_map[annotation['instance_token']] = len(instance_map) + 1
            instance_id = instance_map[annotation['instance_token']]
            instance_attribute = int(annotation['visibility_token'])

            if annotation['instance_token'] not in self.instance_dict:
                # For the first occurrence of an instance
                self.instance_dict[annotation['instance_token']] = {
                    'timestep': [self.counter],
                    'translation': [annotation['translation']],
                    'rotation': [annotation['rotation']],
                    'size': annotation['size'],
                    'instance_id': instance_id,
                    'semantic_id': semantic_id,
                    'attribute_label': [instance_attribute],
                }
            else:
                # For the instance that have appeared before
                self.instance_dict[annotation['instance_token']]['timestep'].append(self.counter)
                self.instance_dict[annotation['instance_token']]['translation'].append(annotation['translation'])
                self.instance_dict[annotation['instance_token']]['rotation'].append(annotation['rotation'])
                self.instance_dict[annotation['instance_token']]['attribute_label'].append(instance_attribute)

        return instance_map

    def get_future_egomotion(self, idx):
        '''
        Calculate LiDAR pose updates between idx and idx+1
        '''
        rec_t0 = self.data_infos[idx]
        future_egomotion = np.eye(4, dtype=np.float32)

        if idx < len(self.data_infos) - 1:
            rec_t1 = self.data_infos[idx + 1]

            if rec_t0['scene_token'] == rec_t1['scene_token']:
                egopose_t0_trans = rec_t0['ego2global_translation']
                egopose_t0_rot = rec_t0['ego2global_rotation']
                egopose_t1_trans = rec_t1['ego2global_translation']
                egopose_t1_rot = rec_t1['ego2global_rotation']
                egopose_t0 = convert_egopose_to_matrix_numpy(egopose_t0_trans, egopose_t0_rot)
                egopose_t1 = convert_egopose_to_matrix_numpy(egopose_t1_trans, egopose_t1_rot)

                lidar2ego_t0_trans = rec_t0['lidar2ego_translation']
                lidar2ego_t0_rot = rec_t0['lidar2ego_rotation']
                lidar2ego_t1_trans = rec_t1['lidar2ego_translation']
                lidar2ego_t1_rot = rec_t1['lidar2ego_rotation']
                lidar2ego_t0 = convert_egopose_to_matrix_numpy(lidar2ego_t0_trans, lidar2ego_t0_rot)
                lidar2ego_t1 = convert_egopose_to_matrix_numpy(lidar2ego_t1_trans, lidar2ego_t1_rot)

                future_egomotion = invert_matrix_egopose_numpy(lidar2ego_t1).dot(invert_matrix_egopose_numpy(egopose_t1)).dot(egopose_t0).dot(lidar2ego_t0)   

        future_egomotion = torch.Tensor(future_egomotion).float()

        # Convert to 6DoF vector
        return future_egomotion.unsqueeze(0)

    @staticmethod
    def _check_consistency(translation, prev_translation, threshold=1.0):
        """
        Check for significant displacement of the instance adjacent moments
        """
        x, y = translation[:2]
        prev_x, prev_y = prev_translation[:2]

        if abs(x - prev_x) > threshold or abs(y - prev_y) > threshold:
            return False
        return True

    def refine_instance_poly(self, instance):
        """
        Fix the missing frames and disturbances of ground truth caused by noise
        """
        pointer = 1
        for i in range(instance['timestep'][0] + 1, self.sequence_length):
            # Fill in the missing frames
            if i not in instance['timestep']:
                instance['timestep'].insert(pointer, i)
                instance['translation'].insert(pointer, instance['translation'][pointer-1])
                instance['rotation'].insert(pointer, instance['rotation'][pointer-1])
                instance['attribute_label'].insert(pointer, instance['attribute_label'][pointer-1])
                pointer += 1
                continue
            
            # Eliminate observation disturbances
            if self._check_consistency(instance['translation'][pointer], instance['translation'][pointer-1]):
                instance['translation'][pointer] = instance['translation'][pointer-1]
                instance['rotation'][pointer] = instance['rotation'][pointer-1]
                instance['attribute_label'][pointer] = instance['attribute_label'][pointer-1]
            pointer += 1
        
        return instance

    def prepare_train_data(self, index):
        '''
        Generate a training sequence
        '''
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        
        example = self.prepare_sequential_data(index)
        return example

    def prepare_test_data(self, index):
        '''
        Generate a test sequence
        TODO: Give additional functions here such as visualization
        '''
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        
        example = self.prepare_sequential_data(index)
        # TODO: visualize example data
        return example
    
    def prepare_sequential_data(self, index):
        '''
        Use the predefined pipeline to generate inputs of the baseline network and ground truth for the standard evaluation protocol in Cam4DOcc
        '''
        instance_map = {}
        input_seq_data = {}
        keys = ['input_dict','future_egomotion', 'sample_token']
        for key in keys:
            input_seq_data[key] = []
        scene_lidar_token = []

        for self.counter, index_t in enumerate(self.indices[index]):
            input_dict_per_frame = self.get_data_info(index_t)
            if input_dict_per_frame is None:
                return None
            
            input_seq_data['input_dict'].append(input_dict_per_frame)
            input_seq_data['sample_token'].append(input_dict_per_frame['sample_idx'])

            instance_map = self.record_instance(index_t, instance_map)
            future_egomotion = self.get_future_egomotion(index_t)
            input_seq_data['future_egomotion'].append(future_egomotion)

            scene_lidar_token.append(input_dict_per_frame['scene_token']+"_"+input_dict_per_frame['lidar_token'])
            if self.counter == self.time_receptive_field - 1:
                self.present_scene_lidar_token = input_dict_per_frame['scene_token']+"_"+input_dict_per_frame['lidar_token']

        # save sequential test indexes for possible evaluation
        if self.test_mode:
            test_idx_path = os.path.join(self.idx_root, "test_ids")
            if not os.path.exists(test_idx_path):
                os.mkdir(test_idx_path)
            np.savez(os.path.join(test_idx_path, self.present_scene_lidar_token), scene_lidar_token)

        for token in self.instance_dict.keys():
            self.instance_dict[token] = self.refine_instance_poly(self.instance_dict[token])

        input_seq_data.update(
            dict(
                time_receptive_field=self.time_receptive_field,
                sequence_length=self.sequence_length,
                egopose_list=self.egopose_list,
                ego2lidar_list=self.ego2lidar_list,
                instance_dict=self.instance_dict,
                instance_map=instance_map,
                indices=self.indices[index],
                scene_token=self.present_scene_lidar_token,
            ))

        example = self.pipeline(input_seq_data)

        return example


    def get_data_info(self, index):
        '''
        get_data_info from .pkl also used by OpenOccupancy
        '''
        
        info = self.data_infos[index]
        
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            lidar2ego_translation=info['lidar2ego_translation'],
            lidar2ego_rotation=info['lidar2ego_rotation'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            # frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
            occ_size = np.array(self.occ_size),
            pc_range = np.array(self.pc_range),
            lidar_token=info['lidar_token'],
            lidarseg=info['lidarseg'],
            curr=info,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            
            lidar2cam_dic = {}
            
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)
                
                lidar2cam_dic[cam_type] = lidar2cam_rt.T

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                    lidar2cam_dic=lidar2cam_dic,
                ))

        return input_dict


    def evaluate(self, results, logger=None, **kawrgs):
        '''
        Evaluate by IOU and VPQ metrics for model evaluation
        '''
        eval_results = {}
        
        ''' calculate IOU '''
        hist_for_iou = sum(results['hist_for_iou'])
        ious = cm_to_ious(hist_for_iou)
        res_table, res_dic = format_iou_results(ious, return_dic=True)
        for key, val in res_dic.items():
            eval_results['IOU_{}'.format(key)] = val
        if logger is not None:
            logger.info('IOU Evaluation')
            logger.info(res_table)        

        ''' calculate VPQ '''
        if 'vpq_metric' in results.keys() and 'vpq_len' in results.keys():
            vpq_sum = sum(results['vpq_metric'])
            eval_results['VPQ'] = vpq_sum/results['vpq_len']

        return eval_results
