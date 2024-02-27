# Developed by Junyi Ma based on the codebase of PowerBEV
# Cam4DOcc: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications
# https://github.com/haomo-ai/Cam4DOcc

import numpy as np
from mmdet.datasets.builder import PIPELINES
import os
import torch
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
import time

@PIPELINES.register_module()
class LoadInstanceWithFlow(object):
    def __init__(self, cam4docc_dataset_path, grid_size=[512, 512, 40], pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], background=0, use_flow=True):
        '''
        Loading sequential occupancy labels and instance flows for training and testing

        cam4docc_dataset_path: data path of Cam4DOcc dataset, including 'segmentation', 'instance', and 'flow'
        grid_size: number of grids along H W L, default: [512, 512, 40]
        pc_range: predefined ranges along H W L, default: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        background: background pixel value for segmentation/instance/flow maps, default: 0
        use_flow: whether use flow for training schemes, default: True
        '''

        self.cam4docc_dataset_path = cam4docc_dataset_path

        self.pc_range = pc_range
        self.resolution = [(self.pc_range[3+i] - self.pc_range[i])/grid_size[i] for i in range(len(self.pc_range[:3]))]
        self.start_position = [self.pc_range[i] + self.resolution[i] / 2.0 for i in range(len(self.pc_range[:3]))]
        self.dimension = grid_size

        self.pc_range = np.array(self.pc_range)
        self.resolution = np.array(self.resolution)
        self.start_position = np.array(self.start_position)
        self.dimension = np.array(self.dimension)

        self.background = background
        self.use_flow = use_flow

    def get_poly_region(self, instance_annotation, present_egopose, present_ego2lidar):
        """
        Obtain the bounding box polygon of the instance
        """
        present_ego_translation, present_ego_rotation = present_egopose
        present_ego2lidar_translation, present_ego2lidar_rotation = present_ego2lidar

        box = Box(
            instance_annotation['translation'], instance_annotation['size'], Quaternion(instance_annotation['rotation'])
        )
        box.translate(present_ego_translation)
        box.rotate(present_ego_rotation)

        box.translate(present_ego2lidar_translation)
        box.rotate(present_ego2lidar_rotation)
        pts=box.corners().T

        X_min_box = pts.min(axis=0)[0]
        X_max_box = pts.max(axis=0)[0]
        Y_min_box = pts.min(axis=0)[1]
        Y_max_box = pts.max(axis=0)[1]
        Z_min_box = pts.min(axis=0)[2]
        Z_max_box = pts.max(axis=0)[2]

        if self.pc_range[0] <= X_min_box and X_max_box <= self.pc_range[3] \
                and self.pc_range[1] <= Y_min_box and Y_max_box <= self.pc_range[4] \
                and self.pc_range[2] <= Z_min_box and Z_max_box <= self.pc_range[5]:
            pts = np.round((pts - self.start_position[:3] + self.resolution[:3] / 2.0) / self.resolution[:3]).astype(np.int32)

            return pts
        else:
            return None

    def fill_occupancy(self, occ_instance, occ_segmentation, occ_attribute_label, instance_fill_info):

        x_grid = torch.linspace(0, self.dimension[0]-1, self.dimension[0], dtype=torch.float)
        x_grid = x_grid.view(self.dimension[0], 1, 1).expand(self.dimension[0], self.dimension[1], self.dimension[2])
        y_grid = torch.linspace(0, self.dimension[1]-1, self.dimension[1], dtype=torch.float)
        y_grid = y_grid.view(1, self.dimension[1], 1).expand(self.dimension[0], self.dimension[1], self.dimension[2])
        z_grid = torch.linspace(0, self.dimension[2]-1, self.dimension[2], dtype=torch.float)
        z_grid = z_grid.view(1, 1, self.dimension[2]).expand(self.dimension[0], self.dimension[1], self.dimension[2])
        mesh_grid_3d = torch.stack((x_grid, y_grid, z_grid), -1)
        mesh_grid_3d = mesh_grid_3d.view(-1, 3)

        occ_instance = torch.from_numpy(occ_instance).view(-1, 1)
        occ_segmentation = torch.from_numpy(occ_segmentation).view(-1, 1)
        occ_attribute_label = torch.from_numpy(occ_attribute_label).view(-1, 1)

        for instance_info in instance_fill_info:
            poly_region_pts = instance_info['poly_region']
            semantic_id = instance_info['semantic_id']
            instance_id = instance_info['instance_id']
            attribute_label=instance_info['attribute_label']

            X_min_box = poly_region_pts.min(axis=0)[0]
            X_max_box = poly_region_pts.max(axis=0)[0]
            Y_min_box = poly_region_pts.min(axis=0)[1]
            Y_max_box = poly_region_pts.max(axis=0)[1]
            Z_min_box = poly_region_pts.min(axis=0)[2]
            Z_max_box = poly_region_pts.max(axis=0)[2]

            mask_cur_instance = (mesh_grid_3d[:,0] >= X_min_box) & (X_max_box >= mesh_grid_3d[:,0]) \
                                & (mesh_grid_3d[:,1] >= Y_min_box) & (Y_max_box >= mesh_grid_3d[:,1]) \
                                & (mesh_grid_3d[:,2] >= Z_min_box) & (Z_max_box >= mesh_grid_3d[:,2])
            occ_instance[mask_cur_instance] = instance_id
            occ_segmentation[mask_cur_instance] = semantic_id
            occ_attribute_label[mask_cur_instance] = attribute_label
        
        occ_instance = occ_instance.view(self.dimension[0], self.dimension[1], self.dimension[2]).long()
        occ_segmentation = occ_segmentation.view(self.dimension[0], self.dimension[1], self.dimension[2]).long()
        occ_attribute_label = occ_attribute_label.view(self.dimension[0], self.dimension[1], self.dimension[2]).long()

        return occ_instance, occ_segmentation, occ_attribute_label


    def get_label(self, input_seq_data):
        """
        Generate labels for semantic segmentation, instance segmentation, z position, attribute from the raw data of nuScenes
        """
        timestep = self.counter
        # Background is ID 0
        segmentation = np.ones((self.dimension[0], self.dimension[1], self.dimension[2])) * self.background
        instance = np.ones((self.dimension[0], self.dimension[1], self.dimension[2])) * self.background
        attribute_label = np.ones((self.dimension[0], self.dimension[1], self.dimension[2]))  * self.background
        
        instance_dict = input_seq_data['instance_dict']
        egopose_list = input_seq_data['egopose_list']
        ego2lidar_list = input_seq_data['ego2lidar_list']
        time_receptive_field = input_seq_data['time_receptive_field']

        instance_fill_info = []
        
        for instance_token, instance_annotation in instance_dict.items():
            if timestep not in instance_annotation['timestep']:
                continue
            pointer = instance_annotation['timestep'].index(timestep)
            annotation = {
                'translation': instance_annotation['translation'][pointer],
                'rotation': instance_annotation['rotation'][pointer],
                'size': instance_annotation['size'],
            }
            
            poly_region = self.get_poly_region(annotation, egopose_list[time_receptive_field - 1], ego2lidar_list[time_receptive_field - 1]) 

            if isinstance(poly_region, np.ndarray):
                if self.counter >= time_receptive_field and instance_token not in self.visible_instance_set:
                    continue
                self.visible_instance_set.add(instance_token)

                prepare_for_fill = dict(
                    poly_region=poly_region,
                    instance_id=instance_annotation['instance_id'],
                    attribute_label=instance_annotation['attribute_label'][pointer],
                    semantic_id=instance_annotation['semantic_id'],
                )

                instance_fill_info.append(prepare_for_fill)

        instance, segmentation, attribute_label = self.fill_occupancy(instance, segmentation, attribute_label, instance_fill_info)

        segmentation = segmentation.unsqueeze(0)
        instance = instance.unsqueeze(0)
        attribute_label = attribute_label.unsqueeze(0).unsqueeze(0)

        return segmentation, instance, attribute_label


    @staticmethod
    def generate_flow(flow, occ_instance_seq, instance, instance_id):
        """
        Generate ground truth for the flow of each instance based on instance segmentation
        """
        seg_len, wx, wy, wz = occ_instance_seq.shape
        ratio = 4
        occ_instance_seq = occ_instance_seq.reshape(seg_len, wx//ratio, ratio, wy//ratio, ratio, wz//ratio, ratio).permute(0,1,3,5,2,4,6).reshape(seg_len, wx//ratio, wy//ratio, wz//ratio, ratio**3)
        empty_mask = occ_instance_seq.sum(-1) == 0
        occ_instance_seq = occ_instance_seq.to(torch.int64)
        occ_space = occ_instance_seq[~empty_mask]
        occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1 
        occ_instance_seq[~empty_mask] = occ_space
        occ_instance_seq = torch.mode(occ_instance_seq, dim=-1)[0]
        occ_instance_seq[occ_instance_seq<0] = 0
        occ_instance_seq = occ_instance_seq.long()

        _, wx, wy, wz = occ_instance_seq.shape 
        x, y, z = torch.meshgrid(torch.arange(wx, dtype=torch.float), torch.arange(wy, dtype=torch.float), torch.arange(wz, dtype=torch.float))
        
        grid = torch.stack((x, y, z), dim=0)

        # Set the first frame
        init_pointer = instance['timestep'][0]
        instance_mask = (occ_instance_seq[init_pointer] == instance_id)

        flow[init_pointer, 0, instance_mask] = grid[0, instance_mask].mean(dim=0, keepdim=True).round() - grid[0, instance_mask]
        flow[init_pointer, 1, instance_mask] = grid[1, instance_mask].mean(dim=0, keepdim=True).round() - grid[1, instance_mask]
        flow[init_pointer, 2, instance_mask] = grid[2, instance_mask].mean(dim=0, keepdim=True).round() - grid[2, instance_mask]

        for i, timestep in enumerate(instance['timestep']):
            if i == 0:
                continue

            instance_mask = (occ_instance_seq[timestep] == instance_id)
            prev_instance_mask = (occ_instance_seq[timestep-1] == instance_id)
            if instance_mask.sum() == 0 or prev_instance_mask.sum() == 0:
                continue

            flow[timestep, 0, instance_mask] = grid[0, prev_instance_mask].mean(dim=0, keepdim=True).round() - grid[0, instance_mask]
            flow[timestep, 1, instance_mask] = grid[1, prev_instance_mask].mean(dim=0, keepdim=True).round() - grid[1, instance_mask]
            flow[timestep, 2, instance_mask] = grid[2, prev_instance_mask].mean(dim=0, keepdim=True).round() - grid[2, instance_mask]

        return flow

    def get_flow_label(self, input_seq_data, ignore_index=255):
        """
        Generate the global map of the flow ground truth
        """
        occ_instance = input_seq_data['instance']
        instance_dict = input_seq_data['instance_dict']
        instance_map = input_seq_data['instance_map']

        seq_len, wx, wy, wz = occ_instance.shape
        ratio = 4
        flow = ignore_index * torch.ones(seq_len, 3, wx//ratio, wy//ratio, wz//ratio)
        
        # ignore flow generation for faster pipelines
        if not self.use_flow:
            return flow

        for token, instance in instance_dict.items():
            flow = self.generate_flow(flow, occ_instance, instance, instance_map[token])
        return flow.float()

    # set ignore index to 0 for vis
    @staticmethod
    def convert_instance_mask_to_center_and_offset_label(input_seq_data, ignore_index=255, sigma=3):
        occ_instance = input_seq_data['instance']
        num_instances=len(input_seq_data['instance_map'])

        seq_len, wx, wy, wz = occ_instance.shape
        center_label = torch.zeros(seq_len, 1, wx, wy, wz)
        offset_label = ignore_index * torch.ones(seq_len, 3, wx, wy, wz)
        # x is vertical displacement, y is horizontal displacement
        x, y, z = torch.meshgrid(torch.arange(wx, dtype=torch.float), torch.arange(wy, dtype=torch.float), torch.arange(wz, dtype=torch.float))
        
        # Ignore id 0 which is the background
        for instance_id in range(1, num_instances+1):
            for t in range(seq_len):
                instance_mask = (occ_instance[t] == instance_id)

                xc = x[instance_mask].mean().round().long()
                yc = y[instance_mask].mean().round().long()
                zc = z[instance_mask].mean().round().long()

                off_x = xc - x
                off_y = yc - y
                off_z = zc - z

                g = torch.exp(-(off_x ** 2 + off_y ** 2 + off_z ** 2) / sigma ** 2)
                center_label[t, 0] = torch.maximum(center_label[t, 0], g)
                offset_label[t, 0, instance_mask] = off_x[instance_mask]
                offset_label[t, 1, instance_mask] = off_y[instance_mask]
                offset_label[t, 2, instance_mask] = off_z[instance_mask]

        return center_label, offset_label

    def __call__(self, results):
        assert 'segmentation' not in results.keys()
        assert 'instance' not in results.keys()
        assert 'attribute_label' not in results.keys()

        time_receptive_field = results['time_receptive_field']
        seg_label_path = os.path.join(self.cam4docc_dataset_path, "segmentation", \
            results['input_dict'][time_receptive_field-1]['scene_token']+"_"+results['input_dict'][time_receptive_field-1]['lidar_token'])
        
        instance_label_path = os.path.join(self.cam4docc_dataset_path, "instance", \
            results['input_dict'][time_receptive_field-1]['scene_token']+"_"+results['input_dict'][time_receptive_field-1]['lidar_token'])
        
        flow_label_path = os.path.join(self.cam4docc_dataset_path, "flow", \
            results['input_dict'][time_receptive_field-1]['scene_token']+"_"+results['input_dict'][time_receptive_field-1]['lidar_token'])

        segmentation_list = []
        if os.path.exists(seg_label_path+".npz"):
            gt_segmentation_arr = np.load(seg_label_path+".npz",allow_pickle=True)['arr_0']
            for j in range(len(gt_segmentation_arr)):
                segmentation = np.zeros((self.dimension[0], self.dimension[1], self.dimension[2])) * self.background
                gt_segmentation = gt_segmentation_arr[j]
                gt_segmentation = torch.from_numpy(gt_segmentation)
                # for i in range(gt_segmentation.shape[0]):
                #     cur_ind = gt_segmentation[i, :3].long()
                #     cur_label = gt_segmentation[i, -1]
                #     segmentation[cur_ind[0],cur_ind[1],cur_ind[2]] = cur_label
                segmentation[gt_segmentation[:, 0].long(), gt_segmentation[:, 1].long(), gt_segmentation[:, 2].long()] = gt_segmentation[:, -1]
                segmentation = torch.from_numpy(segmentation).unsqueeze(0)
                segmentation_list.append(segmentation)

        instance_list = []
        if os.path.exists(instance_label_path+".npz"):
            gt_instance_arr = np.load(instance_label_path+".npz",allow_pickle=True)['arr_0']

            for j in range(len(gt_instance_arr)):
                instance = np.ones((self.dimension[0], self.dimension[1], self.dimension[2])) * self.background
                gt_instance = gt_instance_arr[j]
                gt_instance = torch.from_numpy(gt_instance)
                # for i in range(gt_instance.shape[0]):
                #     cur_ind = gt_instance[i, :3].long()
                #     cur_label = gt_instance[i, -1]
                #     instance[cur_ind[0],cur_ind[1],cur_ind[2]] = cur_label
                instance[gt_instance[:, 0].long(), gt_instance[:, 1].long(), gt_instance[:, 2].long()] = gt_instance[:, -1]
                instance = torch.from_numpy(instance).unsqueeze(0)
                instance_list.append(instance)
        
        flow_list = []
        if os.path.exists(flow_label_path+".npz"):
            gt_flow_arr = np.load(flow_label_path+".npz",allow_pickle=True)['arr_0']

            for j in range(len(gt_flow_arr)):
                flow = np.ones((3, self.dimension[0]//4, self.dimension[1]//4, self.dimension[2]//4)) * 255
                gt_flow = gt_flow_arr[j]
                gt_flow = torch.from_numpy(gt_flow)
                # for i in range(gt_flow.shape[0]):
                #     cur_ind = gt_flow[i, :3].long()
                #     cur_label = gt_flow[i, 3:]
                #     flow[0, cur_ind[0],cur_ind[1],cur_ind[2]] = cur_label[0]
                #     flow[1, cur_ind[0],cur_ind[1],cur_ind[2]] = cur_label[1]
                #     flow[2, cur_ind[0],cur_ind[1],cur_ind[2]] = cur_label[2]
                flow[:, gt_flow[:, 0].long(), gt_flow[:, 1].long(), gt_flow[:, 2].long()] = gt_flow[:, 3:].permute(1, 0)
                flow = torch.from_numpy(flow).unsqueeze(0)
                flow_list.append(flow)


        if os.path.exists(seg_label_path+".npz") and os.path.exists(instance_label_path+".npz") and os.path.exists(flow_label_path+".npz"):
            results['segmentation'] = torch.cat(segmentation_list, dim=0)
            results['instance'] = torch.cat(instance_list, dim=0)
            results['attribute_label'] =  torch.from_numpy(np.zeros((self.dimension[0], self.dimension[1], self.dimension[2]))).unsqueeze(0)
            results['flow'] = torch.cat(flow_list, dim=0).float()

            for key, value in results.items():
                if key in ['sample_token', 'centerness', 'offset', 'flow', 'time_receptive_field', "indices", \
                  'segmentation','instance','attribute_label','sequence_length', 'instance_dict', 'instance_map', 'input_dict', 'egopose_list','ego2lidar_list','scene_token']:
                    continue
                results[key] = torch.cat(value, dim=0)
            return results

        else:
            results['segmentation'] = []
            results['instance'] = []
            results['attribute_label'] = []

            segmentation_saved_list = []
            instance_saved_list = []

            sequence_length = results['sequence_length']
            self.visible_instance_set = set()
            for self.counter in range(sequence_length):
                segmentation, instance, attribute_label = self.get_label(results)
                results['segmentation'].append(segmentation)
                results['instance'].append(instance)
                results['attribute_label'].append(attribute_label)

                x_grid = torch.linspace(0, self.dimension[0]-1, self.dimension[0], dtype=torch.long)
                x_grid = x_grid.view(self.dimension[0], 1, 1).expand(self.dimension[0], self.dimension[1], self.dimension[2])
                y_grid = torch.linspace(0, self.dimension[1]-1, self.dimension[1], dtype=torch.long)
                y_grid = y_grid.view(1, self.dimension[1], 1).expand(self.dimension[0], self.dimension[1], self.dimension[2])
                z_grid = torch.linspace(0, self.dimension[2]-1, self.dimension[2], dtype=torch.long)
                z_grid = z_grid.view(1, 1, self.dimension[2]).expand(self.dimension[0], self.dimension[1], self.dimension[2])
                segmentation_for_save = torch.stack((x_grid, y_grid, z_grid), -1)
                segmentation_for_save = segmentation_for_save.view(-1, 3)
                segmentation_label = segmentation.squeeze(0).view(-1,1)
                segmentation_for_save = torch.cat((segmentation_for_save, segmentation_label), dim=-1)
                kept = segmentation_for_save[:,-1]!=0
                segmentation_for_save= segmentation_for_save[kept]
                segmentation_saved_list.append(segmentation_for_save)


                x_grid = torch.linspace(0, self.dimension[0]-1, self.dimension[0], dtype=torch.long)
                x_grid = x_grid.view(self.dimension[0], 1, 1).expand(self.dimension[0], self.dimension[1], self.dimension[2])
                y_grid = torch.linspace(0, self.dimension[1]-1, self.dimension[1], dtype=torch.long)
                y_grid = y_grid.view(1, self.dimension[1], 1).expand(self.dimension[0], self.dimension[1], self.dimension[2])
                z_grid = torch.linspace(0, self.dimension[2]-1, self.dimension[2], dtype=torch.long)
                z_grid = z_grid.view(1, 1, self.dimension[2]).expand(self.dimension[0], self.dimension[1], self.dimension[2])
                instance_for_save = torch.stack((x_grid, y_grid, z_grid), -1)
                instance_for_save = instance_for_save.view(-1, 3)
                instance_label = instance.squeeze(0).view(-1,1)
                instance_for_save = torch.cat((instance_for_save, instance_label), dim=-1)
                kept = instance_for_save[:,-1]!=0
                instance_for_save= instance_for_save[kept]
                instance_saved_list.append(instance_for_save)
            
            segmentation_saved_list2 = [item.cpu().detach().numpy() for item in segmentation_saved_list]
            instance_saved_list2 = [item.cpu().detach().numpy() for item in instance_saved_list]

            np.savez(seg_label_path, segmentation_saved_list2)
            np.savez(instance_label_path, instance_saved_list2)

            results['segmentation'] = torch.cat(results['segmentation'], dim=0)
            results['instance'] = torch.cat(results['instance'], dim=0)
            results['attribute_label'] =  torch.from_numpy(np.zeros((self.dimension[0], self.dimension[1], self.dimension[2]))).unsqueeze(0)

            results['flow'] = self.get_flow_label(results, ignore_index=255)
            
            flow_saved_list = []
            sequence_length = results['sequence_length']
            d0 = self.dimension[0]//4
            d1 = self.dimension[1]//4 
            d2 = self.dimension[2]//4 
            for cnt in range(sequence_length):
                flow = results['flow'][cnt, ...]
                x_grid = torch.linspace(0, d0-1, d0, dtype=torch.long)
                x_grid = x_grid.view(d0, 1, 1).expand(d0, d1, d2)
                y_grid = torch.linspace(0, d1-1, d1, dtype=torch.long)
                y_grid = y_grid.view(1, d1, 1).expand(d0, d1, d2)
                z_grid = torch.linspace(0, d2-1, d2, dtype=torch.long)
                z_grid = z_grid.view(1, 1, d2).expand(d0, d1, d2)
                flow_for_save = torch.stack((x_grid, y_grid, z_grid), -1)
                flow_for_save = flow_for_save.view(-1, 3)
                flow_label = flow.permute(1,2,3,0).view(-1,3)
                flow_for_save = torch.cat((flow_for_save, flow_label), dim=-1)
                kept = (flow_for_save[:,-1]!=255) & (flow_for_save[:,-2]!=255) & (flow_for_save[:,-3]!=255)
                flow_for_save= flow_for_save[kept]
                flow_saved_list.append(flow_for_save)

            flow_saved_list2 = [item.cpu().detach().numpy() for item in flow_saved_list]
            np.savez(flow_label_path, flow_saved_list2)

            for key, value in results.items():
                if key in ['sample_token', 'centerness', 'offset', 'flow', 'time_receptive_field', "indices", \
                   'segmentation','instance','attribute_label','sequence_length', 'instance_dict', 'instance_map', 'input_dict', 'egopose_list','ego2lidar_list','scene_token']:
                    continue
                results[key] = torch.cat(value, dim=0)

        return results