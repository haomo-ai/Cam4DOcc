# Developed by Junyi Ma based on the codebase of OpenOccupancy and PowerBEV
# Cam4DOcc: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications
# https://github.com/haomo-ai/Cam4DOcc

import numpy as np
from mmcv.runner import get_dist_info
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.datasets.pipelines import Compose
from torch.utils.data import Dataset
from lyft_dataset_sdk.lyftdataset import LyftDataset
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
class Cam4DOccLyftDataset(Dataset):
    def __init__(self, occ_size, pc_range, occ_root, idx_root, time_receptive_field, n_future_frames, classes, use_separate_classes,
                  train_capacity, test_capacity, test_mode=False, pipeline=None, **kwargs):
        
        '''
        Cam4DOccLyftDataset contains sequential occupancy states as well as instance flow for training occupancy forecasting models. We unify the related operations in the LiDAR coordinate system following OpenOccupancy.

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

        self.test_mode = test_mode
        self.CLASSES = classes
        
        self.train_capacity = train_capacity
        self.test_capacity = test_capacity

        super().__init__()

        # training and test indexes following PowerBEV
        self.TRAIN_LYFT_INDICES = [1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16,
                      17, 18, 19, 20, 21, 23, 24, 27, 28, 29, 30, 31, 32,
                      33, 35, 36, 37, 39, 41, 43, 44, 45, 46, 47, 48, 49,
                      50, 51, 52, 53, 55, 56, 59, 60, 62, 63, 65, 68, 69,
                      70, 71, 72, 73, 74, 75, 76, 78, 79, 81, 82, 83, 84,
                      86, 87, 88, 89, 93, 95, 97, 98, 99, 103, 104, 107, 108,
                      109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 121, 122, 124,
                      127, 128, 130, 131, 132, 134, 135, 136, 137, 138, 139, 143, 144,
                      146, 147, 148, 149, 150, 151, 152, 153, 154, 156, 157, 158, 159,
                      161, 162, 165, 166, 167, 171, 172, 173, 174, 175, 176, 177, 178,
                      179]

        self.VAL_LYFT_INDICES = [0, 2, 4, 13, 22, 25, 26, 34, 38, 40, 42, 54, 57,
                    58, 61, 64, 66, 67, 77, 80, 85, 90, 91, 92, 94, 96,
                    100, 101, 102, 105, 106, 112, 120, 123, 125, 126, 129, 133, 140,
                    141, 142, 145, 155, 160, 163, 164, 168, 169, 170]


        rank, world_size = get_dist_info()

        self.time_receptive_field = time_receptive_field
        self.n_future_frames = n_future_frames
        self.sequence_length = time_receptive_field + n_future_frames

        if rank == 0:
            print("-------------")
            print("use past " + str(self.time_receptive_field) + " frames to forecast future " + str(self.n_future_frames) + " frames")
            print("-------------")

        self.occ_size = occ_size
        self.pc_range = pc_range
        self.occ_root = occ_root
        self.idx_root = idx_root
        self.classes = classes
        self.use_separate_classes = use_separate_classes

        self.pipeline = Compose(pipeline)

        # load origin nusc dataset for instance annotation
        self.lyft = LyftDataset(data_path='/tos://haomo-algorithms/c6089dc67ff976615510d22b5eaaaa4e/mjy/cam4docc/data/lyft/', 
                                json_path='/tos://haomo-algorithms/c6089dc67ff976615510d22b5eaaaa4e/mjy/cam4docc/data/lyft/train_data', verbose=False)

        self.scenes = self.get_scenes()
        self.ixes = self.get_samples()
        self.indices = self.get_indices()

        self.present_scene_lidar_token = " "
        self._set_group_flag()

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

    def get_scenes(self):
        """
        Obtain the list of scenes names in the given split.
        """
        scenes = [row['name'] for row in self.lyft.scene]
        # split in train/val
        indices = self.VAL_LYFT_INDICES  if self.test_mode else  self.TRAIN_LYFT_INDICES
        scenes = [scenes[i] for i in indices]
        return scenes

    def get_samples(self):
        """
        Find and sort the samples in the given split by scene.
        """
        samples = [sample for sample in self.lyft.sample]
        # remove samples that aren't in this split
        samples = [sample for sample in samples if self.lyft.get('scene', sample['scene_token'])['name'] in self.scenes]
        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def get_indices(self):
        '''
        Generate sequential indexes for training and testing
        '''
        indices = []
        for index in range(len(self.ixes)):
            is_valid_data = True
            previous_rec = None
            current_indices = []
            for t in range(self.sequence_length):
                index_t = index + t
                # Going over the dataset size limit.
                if index_t >= len(self.ixes):
                    is_valid_data = False
                    break
                rec = self.ixes[index_t]
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
        current_sample = self.lyft.get('sample', rec['token'])
        egopose = self.lyft.get('ego_pose', self.lyft.get('sample_data', current_sample['data']['LIDAR_TOP'])['ego_pose_token'])
        ego2global_translation = egopose['translation']
        ego2global_rotation = egopose['rotation']
        trans = -np.array(ego2global_translation)
        rot = Quaternion(ego2global_rotation).inverse
        
        return trans, rot
    
    def get_ego2lidar_pose(self, rec):
        '''
        Get LiDAR poses in ego system
        '''
        current_sample = self.lyft.get('sample', rec['token'])
        lidar_top_data = self.lyft.get('sample_data', current_sample['data']['LIDAR_TOP'])
        lidar2ego_translation = self.lyft.get('calibrated_sensor', lidar_top_data['calibrated_sensor_token'])['translation']
        lidar2ego_rotation =  self.lyft.get('calibrated_sensor', lidar_top_data['calibrated_sensor_token'])['rotation']

        trans = -np.array(lidar2ego_translation)
        rot = Quaternion(lidar2ego_rotation).inverse
        return trans, rot

    def record_instance(self, idx, instance_map):
        """
        Record information about each visible instance in the sequence and assign a unique ID to it
        """
        rec = self.ixes[idx]
        translation, rotation = self.get_lidar_pose(rec)
        self.egopose_list.append([translation, rotation])
        ego2lidar_translation, ego2lidar_rotation = self.get_ego2lidar_pose(rec)
        self.ego2lidar_list.append([ego2lidar_translation, ego2lidar_rotation])

        current_sample = self.lyft.get('sample', rec['token'])
        for annotation_token in current_sample['anns']:
            annotation = self.lyft.get('sample_annotation', annotation_token)
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

            if annotation['instance_token'] not in instance_map:
                instance_map[annotation['instance_token']] = len(instance_map) + 1
            instance_id = instance_map[annotation['instance_token']]
            instance_attribute = 1 # deprecated

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
        rec_t0 = self.ixes[idx]
        future_egomotion = np.eye(4, dtype=np.float32)

        if idx < len(self.ixes) - 1:
            rec_t1 = self.ixes[idx + 1]

            if rec_t0['scene_token'] == rec_t1['scene_token']:
                egopose_t0 = self.lyft.get('ego_pose', self.lyft.get('sample_data', rec_t0['data']['LIDAR_TOP'])['ego_pose_token'])
                egopose_t0_trans = egopose_t0['translation']
                egopose_t0_rot = egopose_t0['rotation']

                egopose_t1 = self.lyft.get('ego_pose', self.lyft.get('sample_data', rec_t1['data']['LIDAR_TOP'])['ego_pose_token'])
                egopose_t1_trans = egopose_t1['translation']
                egopose_t1_rot = egopose_t1['rotation']

                egopose_t0 = convert_egopose_to_matrix_numpy(egopose_t0_trans, egopose_t0_rot)
                egopose_t1 = convert_egopose_to_matrix_numpy(egopose_t1_trans, egopose_t1_rot)

                lidar_top_data_t0 = self.lyft.get('sample_data', rec_t0['data']['LIDAR_TOP'])
                lidar2ego_t0_trans = self.lyft.get('calibrated_sensor', lidar_top_data_t0['calibrated_sensor_token'])['translation']
                lidar2ego_t0_rot =  self.lyft.get('calibrated_sensor', lidar_top_data_t0['calibrated_sensor_token'])['rotation']
                lidar_top_data_t1 = self.lyft.get('sample_data', rec_t1['data']['LIDAR_TOP'])
                lidar2ego_t1_trans = self.lyft.get('calibrated_sensor', lidar_top_data_t1['calibrated_sensor_token'])['translation']
                lidar2ego_t1_rot =  self.lyft.get('calibrated_sensor', lidar_top_data_t1['calibrated_sensor_token'])['rotation']


                lidar2ego_t0 = convert_egopose_to_matrix_numpy(lidar2ego_t0_trans, lidar2ego_t0_rot)
                lidar2ego_t1 = convert_egopose_to_matrix_numpy(lidar2ego_t1_trans, lidar2ego_t1_rot)

                future_egomotion = invert_matrix_egopose_numpy(lidar2ego_t1).dot(invert_matrix_egopose_numpy(egopose_t1)).dot(egopose_t0).dot(lidar2ego_t0)   

        future_egomotion = torch.Tensor(future_egomotion).float()
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
        
        example = self.prepare_sequential_data(index)
        return example

    def prepare_test_data(self, index):
        '''
        Generate a test sequence
        TODO: Give additional functions here such as visualization
        '''
        
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

            input_dict_per_frame = {}
            rec = self.ixes[index_t]  # sample

            lidar_top_data = self.lyft.get('sample_data', rec['data']['LIDAR_TOP'])
            lidar2ego_translation = self.lyft.get('calibrated_sensor', lidar_top_data['calibrated_sensor_token'])['translation']
            lidar2ego_rotation =  self.lyft.get('calibrated_sensor', lidar_top_data['calibrated_sensor_token'])['rotation']

            egopose = self.lyft.get('ego_pose', self.lyft.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
            ego2global_translation = egopose['translation']
            ego2global_rotation = egopose['rotation']

            input_dict_per_frame['lidar2ego_translation'] = lidar2ego_translation
            input_dict_per_frame['lidar2ego_rotation'] = lidar2ego_rotation
            input_dict_per_frame['ego2global_translation'] = ego2global_translation
            input_dict_per_frame['ego2global_rotation'] = ego2global_rotation
            input_dict_per_frame['scene_token'] = rec['scene_token']
            input_dict_per_frame['lidar_token'] = rec['data']['LIDAR_TOP']
            input_dict_per_frame['occ_size'] = np.array(self.occ_size)
            input_dict_per_frame['pc_range'] = np.array(self.pc_range)
            input_dict_per_frame['sample_idx'] = rec['token']


            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            cam_intrinsics_ori = []
            lidar2cam_dic = {}

            lidar_sample = self.lyft.get('sample_data', rec['data']['LIDAR_TOP'])
            lidar_pose = self.lyft.get('ego_pose', lidar_sample['ego_pose_token'])
            lidar_rotation = Quaternion(lidar_pose['rotation'])
            lidar_translation = np.array(lidar_pose['translation'])[:, None]
            lidar_to_world = np.vstack([
                np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
                np.array([0, 0, 0, 1])
            ])

            lidar_sample_calib = self.lyft.get('calibrated_sensor', lidar_sample['calibrated_sensor_token'])
            lidar_sensor_rotation = Quaternion(lidar_sample_calib['rotation'])
            lidar_sensor_translation = np.array(lidar_sample_calib['translation'])[:, None]
            lidar_to_lidarego = np.vstack([
                np.hstack((lidar_sensor_rotation.rotation_matrix, lidar_sensor_translation)),
                np.array([0, 0, 0, 1])
            ])

            cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

            for cam in cameras:
                camera_sample = self.lyft.get('sample_data', rec['data'][cam])
                image_paths.append(os.path.join("/tos://haomo-algorithms/c6089dc67ff976615510d22b5eaaaa4e/mjy/cam4docc/data/lyft/", camera_sample['filename']))

                car_egopose = self.lyft.get('ego_pose', camera_sample['ego_pose_token'])
                egopose_rotation = Quaternion(car_egopose['rotation']).inverse
                egopose_translation = -np.array(car_egopose['translation'])[:, None]
                world_to_car_egopose = np.vstack([
                    np.hstack((egopose_rotation.rotation_matrix, egopose_rotation.rotation_matrix @ egopose_translation)),
                    np.array([0, 0, 0, 1])
                ])

                sensor_sample = self.lyft.get('calibrated_sensor', camera_sample['calibrated_sensor_token'])
                intrinsic = torch.Tensor(sensor_sample['camera_intrinsic'])
                cam_intrinsics_ori.append(intrinsic)
                sensor_rotation = Quaternion(sensor_sample['rotation'])
                sensor_translation = np.array(sensor_sample['translation'])[:, None]
                car_egopose_to_sensor = np.vstack([
                    np.hstack((sensor_rotation.rotation_matrix, sensor_translation)),
                    np.array([0, 0, 0, 1])
                ])
                car_egopose_to_sensor = np.linalg.inv(car_egopose_to_sensor)

                lidar_to_sensor = car_egopose_to_sensor @ world_to_car_egopose @ lidar_to_world @ lidar_to_lidarego
                sensor_to_lidar =np.linalg.inv(lidar_to_sensor)

                lidar2cam_r = lidar_to_sensor[:3, :3] 
                lidar2cam_t = sensor_to_lidar[:3, -1].reshape(1,3) @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)
                
                lidar2cam_dic[cam] = lidar2cam_rt.T

            input_dict_per_frame.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    cam_intrinsics=cam_intrinsics_ori,
                    lidar2cam=lidar2cam_rts,
                    lidar2cam_dic=lidar2cam_dic,
                ))
                        
            input_seq_data['input_dict'].append(input_dict_per_frame)

            instance_map = self.record_instance(index_t, instance_map)
            future_egomotion = self.get_future_egomotion(index_t)
            input_seq_data['future_egomotion'].append(future_egomotion)
            input_seq_data['sample_token'].append(input_dict_per_frame['sample_idx'])

            scene_lidar_token.append(input_dict_per_frame['scene_token']+"_"+input_dict_per_frame['lidar_token'])
            if self.counter == self.time_receptive_field - 1:
                self.present_scene_lidar_token = input_dict_per_frame['scene_token']+"_"+input_dict_per_frame['lidar_token']

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
