# Cam4DOcc: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications
# https://github.com/haomo-ai/Cam4DOcc
# 2 classes: inflated GMO and others

# Basic params ******************************************
_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
find_unused_parameters = True
# whether training and test together with dataset generation
only_generate_dataset = False
# we only consider use_camera in Cam4DOcc in the current version
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
plugin = True
plugin_dir = "projects/occ_plugin/"

# path unused for lyft
occ_path = " "
depth_gt_path = " "
train_ann_file = " "
val_ann_file = " "

cam4docc_dataset_path = "./data/cam4docc/"
nusc_root = './data/lyft/'
# GMO class names
class_names = ['vehicle', 'human']
use_separate_classes = False
use_fine_occ = False

# Forecasting-related params ******************************************
# we use *time_receptive_field* past frames to forecast future *n_future_frames* frames
# for 3D instance prediction, n_future_frames_plus > n_future_frames has to be set
time_receptive_field = 3
n_future_frames = 4
n_future_frames_plus = 6
iou_thresh_for_vpq = 0.2
test_present = False

# Occupancy-related params ******************************************
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
occ_size = [512, 512, 40]
lss_downsample = [4, 4, 4]
voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
empty_idx = 0
if use_separate_classes:
    num_cls = len(class_names) + 1
else:
    num_cls = 2
img_norm_cfg = None

# Save params ******************************************
save_pred = False
save_path = "./data/cam4docc/results"

# Data-generation and pipeline params ******************************************
dataset_type = 'Cam4DOccLyftDataset'
file_client_args = dict(backend='disk')
data_config={
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (896, 1600),
    'src_size': (900, 1600),
    # image-view augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

bda_aug_conf = dict(
            rot_lim=(-0, 0),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5)

train_capacity = 15720 # default: use all sequences
test_capacity = 5880 # default: use all sequences
train_pipeline = [
    dict(type='LoadInstanceWithFlow', cam4docc_dataset_path=cam4docc_dataset_path, grid_size=occ_size, use_flow=True, background=empty_idx, pc_range=point_cloud_range,
                use_separate_classes=use_separate_classes, use_lyft=True),
    dict(type='LoadMultiViewImageFromFiles_BEVDet', is_train=True, data_config=data_config,
                sequential=False, aligned=True, trans_only=False, depth_gt_path=depth_gt_path,
                mmlabnorm=True, load_depth=True, img_norm_cfg=img_norm_cfg, use_lyft=True),
    dict(type='LoadOccupancy', to_float32=True, occ_path=occ_path, grid_size=occ_size, unoccupied=empty_idx, pc_range=point_cloud_range, use_fine_occ=use_fine_occ, test_mode=False),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs_seq', 'gt_occ', 'segmentation', 'instance', 'flow', 'future_egomotion']),
]

test_pipeline = [
    dict(type='LoadInstanceWithFlow', cam4docc_dataset_path=cam4docc_dataset_path, grid_size=occ_size, use_flow=True, background=empty_idx, pc_range=point_cloud_range,
         use_separate_classes=use_separate_classes, use_lyft=True),
    dict(type='LoadMultiViewImageFromFiles_BEVDet', data_config=data_config, depth_gt_path=depth_gt_path,
         sequential=False, aligned=True, trans_only=False, mmlabnorm=True, img_norm_cfg=img_norm_cfg, test_mode=True, use_lyft=True),
    dict(type='LoadOccupancy', to_float32=True, occ_path=occ_path, grid_size=occ_size, unoccupied=empty_idx, pc_range=point_cloud_range, use_fine_occ=use_fine_occ, test_mode=True),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names, with_label=False), 
    dict(type='Collect3D', keys=['img_inputs_seq', 'gt_occ', 'segmentation', 'instance', 'flow', 'future_egomotion'], meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token']),
]

train_config=dict(
        type=dataset_type,
        data_root=nusc_root,
        occ_root=occ_path,
        idx_root=cam4docc_dataset_path,
        ori_data_root=cam4docc_dataset_path,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        classes=class_names,
        use_separate_classes=use_separate_classes,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        box_type_3d='LiDAR',
        time_receptive_field=time_receptive_field,
        n_future_frames=n_future_frames,
        train_capacity=train_capacity,
        test_capacity=test_capacity,
        ) 

test_config=dict(
    type=dataset_type,
    occ_root=occ_path,
    data_root=nusc_root,
    idx_root=cam4docc_dataset_path,
    ori_data_root=cam4docc_dataset_path,
    ann_file=val_ann_file,
    pipeline=test_pipeline,
    classes=class_names,
    use_separate_classes=use_separate_classes,
    modality=input_modality,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    time_receptive_field=time_receptive_field, 
    n_future_frames=n_future_frames,
    train_capacity=train_capacity,
    test_capacity=test_capacity,
    ) 

# in our work we use 8 NVIDIA A100 GPUs
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=train_config,
    val=test_config,
    test=test_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

# Model params ******************************************
grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x*lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y*lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z*lss_downsample[2]],
    'dbound': [2.0, 58.0, 0.5],
}

voxel_channels = [32*(time_receptive_field), 32*2*(time_receptive_field), 32*4*(time_receptive_field), 32*8*(time_receptive_field)]
pred_channels = [32, 32*2, 32*4, 32*8]
decoder_channels = [32*(n_future_frames_plus), 32*2*(n_future_frames_plus), 32*4*(n_future_frames_plus), 32*8*(n_future_frames_plus)]

numC_Trans = 64
occ_encoder_input_channel = (numC_Trans+6)*time_receptive_field
voxel_out_channel = 32*(n_future_frames_plus)
flow_out_channel = 32*(n_future_frames_plus)
voxel_out_channel_per_frame = 32
voxel_out_indices = (0, 1, 2, 3)
my_voxel_out_indices = (0, 1, 2, 3)

model = dict(
    type='OCFNet',
    only_generate_dataset=only_generate_dataset,
    loss_norm=False,
    disable_loss_depth=True,
    point_cloud_range=point_cloud_range,
    time_receptive_field=time_receptive_field,
    n_future_frames=n_future_frames,
    n_future_frames_plus=n_future_frames_plus,
    max_label=num_cls,
    iou_thresh_for_vpq=iou_thresh_for_vpq,
    test_present=test_present,
    record_time=False,
    save_pred=save_pred,
    save_path=save_path,
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=0,
        with_cp=False,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128]),
    img_view_transformer=dict(type='ViewTransformerLiftSplatShootVoxel',
                              norm_cfg=dict(type='SyncBN', requires_grad=True),
                              loss_depth_weight=3.,
                              loss_depth_type='kld',
                              grid_config=grid_config,
                              data_config=data_config,
                              numC_Trans=numC_Trans,
                              vp_megvii=False),
    occ_encoder_backbone=dict(
        type='CustomResNet3D',
        depth=18,
        n_input_channels=occ_encoder_input_channel,
        block_inplanes=voxel_channels,
        out_indices=my_voxel_out_indices,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    occ_predictor=dict(
        type='Predictor',
        n_input_channels=pred_channels,
        in_timesteps=time_receptive_field,
        out_timesteps=n_future_frames_plus,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    occ_encoder_neck=dict(
        type='FPN3D',
        with_cp=False,
        in_channels=decoder_channels,
        out_channels=voxel_out_channel,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    flow_encoder_backbone=dict(
        type='CustomResNet3D',
        depth=18,
        n_input_channels=occ_encoder_input_channel,
        block_inplanes=voxel_channels,
        out_indices=my_voxel_out_indices,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    flow_predictor=dict(
        type='Predictor',
        n_input_channels=pred_channels,
        in_timesteps=time_receptive_field,
        out_timesteps=n_future_frames_plus,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    flow_encoder_neck=dict(
        type='FPN3D',
        with_cp=False,
        in_channels=decoder_channels,
        out_channels=flow_out_channel,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    flow_head=dict(
        type='FlowHead',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        soft_weights=True, 
        final_occ_size=occ_size,
        fine_topk=15000,
        empty_idx=empty_idx,
        num_level=len(my_voxel_out_indices),
        in_channels=[voxel_out_channel_per_frame] * len(my_voxel_out_indices),
        out_channel=3,  # 3-dim flow
        point_cloud_range=point_cloud_range,
    ),
    pts_bbox_head=dict(
        type='OccHead',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        soft_weights=True,
        final_occ_size=occ_size,
        fine_topk=15000,
        empty_idx=empty_idx,
        num_level=len(my_voxel_out_indices),
        in_channels=[voxel_out_channel_per_frame] * len(my_voxel_out_indices),
        out_channel=num_cls,
        point_cloud_range=point_cloud_range,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0,
        ),
    ),
    empty_idx=empty_idx,
)

# Learning policy params ******************************************
optimizer = dict(
    type='AdamW',
    lr=3e-4,   
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

runner = dict(type='EpochBasedRunner', max_epochs=24)
evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='SSC_mean',
    rule='greater',
)

custom_hooks = [
    dict(type='OccEfficiencyHook'),
]