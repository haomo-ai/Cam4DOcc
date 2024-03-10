# Cam4DOcc

The official code an data for the benchmark with baselines for our paper: [Cam4DOcc: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications](https://arxiv.org/abs/2311.17663)

This work has been accepted by CVPR 2024 :tada:

[Junyi Ma#](https://github.com/BIT-MJY), [Xieyuanli Chen#](https://github.com/Chen-Xieyuanli), Jiawei Huang, [Jingyi Xu](https://github.com/BIT-XJY), [Zhen Luo](https://github.com/Blurryface0814), Jintao Xu, Weihao Gu, Rui Ai, [Hesheng Wang*](https://scholar.google.com/citations?hl=en&user=q6AY9XsAAAAJ)

<img src="https://github.com/haomo-ai/Cam4DOcc/blob/main/benchmark.png" width="49%"/> <img src="https://github.com/haomo-ai/Cam4DOcc/blob/main/OCFNet.png" width="49%"/>



## Citation
If you use Cam4DOcc in an academic work, please cite our paper:

	@inproceedings{ma2024cvpr,
		author = {Junyi Ma and Xieyuanli Chen and Jiawei Huang and Jingyi Xu and Zhen Luo and Jintao Xu and Weihao Gu and Rui Ai and Hesheng Wang},
		title = {{Cam4DOcc: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications}},
		booktitle = {Proc.~of the IEEE/CVF Conf.~on Computer Vision and Pattern Recognition (CVPR)},
		year = 2024
	}
 
## Installation

<details>
	
<summary>We follow the installation instructions of our codebase OpenOccupancy, which are also posted here
</summary>

* Create a conda virtual environment and activate it
```bash
conda create -n cam4docc python=3.7 -y
conda activate cam4docc
```
* Install PyTorch and torchvision (tested on torch==1.10.1 & cuda=11.3)
```bash
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```
* Install gcc>=5 in conda env
```bash
conda install -c omgarcia gcc-6
```
* Install mmcv, mmdet, and mmseg
```bash
pip install mmcv-full==1.4.0
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```
* Install mmdet3d from the source code
```bash
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install
```
* Install other dependencies
```bash
pip install timm
pip install open3d-python
pip install PyMCubes
pip install spconv-cu113
pip install fvcore
pip install setuptools==59.5.0

pip install lyft_dataset_sdk # for lyft dataset
```
* Install occupancy pooling
```
git clone git@github.com:haomo-ai/Cam4DOcc.git
cd Cam4DOcc
export PYTHONPATH=“.”
python setup.py develop
```

</details>

## Data Structure

### nuScenes dataset
* Please link your [nuScenes V1.0 full dataset](https://www.nuscenes.org/nuscenes#download) to the data folder. 
* [nuScenes-Occupancy](https://drive.google.com/file/d/1vTbgddMzUN6nLyWSsCZMb9KwihS7nPoH/view?usp=sharing), [nuscenes_occ_infos_train.pkl](https://github.com/JeffWang987/OpenOccupancy/releases/tag/train_pkl), and [nuscenes_occ_infos_val.pkl](https://github.com/JeffWang987/OpenOccupancy/releases/tag/val_pkl) are also provided by the previous work. If you only want to reproduce the forecasting results with "inflated" form, nuScenes dataset and Cam4DOcc are all you need.

### Lyft dataset
* Please link your [Lyft dataset](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/data) to the data folder.
* The required folders are listed below.

Note that the folders under `cam4docc` will be generated automatically once you first run our training or evaluation scripts.

```bash
Cam4DOcc
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── lidarseg/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
│   │   ├── nuscenes_occ_infos_train.pkl
│   │   ├── nuscenes_occ_infos_val.pkl
│   ├── nuScenes-Occupancy/
│   ├── lyft/
│   │   ├── maps/
│   │   ├── train_data/
│   │   ├── images/   # from train images, containing xxx.jpeg
│   ├── cam4docc
│   │   ├── GMO/
│   │   │   ├── segmentation/
│   │   │   ├── instance/
│   │   │   ├── flow/
│   │   ├── MMO/
│   │   │   ├── segmentation/
│   │   │   ├── instance/
│   │   │   ├── flow/
│   │   ├── GMO_lyft/
│   │   │   ├── ...
│   │   ├── MMO_lyft/
│   │   │   ├── ...
```
Alternatively, you could manually modify the path parameters in the [config files](https://github.com/haomo-ai/Cam4DOcc/tree/main/projects/configs/baselines) instead of using the default data structure, which are also listed here:
```
occ_path = "./data/nuScenes-Occupancy"
depth_gt_path = './data/depth_gt'
train_ann_file = "./data/nuscenes/nuscenes_occ_infos_train.pkl"
val_ann_file = "./data/nuscenes/nuscenes_occ_infos_val.pkl"
cam4docc_dataset_path = "./data/cam4docc/"
nusc_root = './data/nuscenes/'
```

## Training and Evaluation

We directly integrate the Cam4DOcc dataset generation pipeline into the dataloader, so you can directly run training or evaluate scripts and just wait :smirk:

Optionally, you can set `only_generate_dataset=True` in the [config files](https://github.com/haomo-ai/Cam4DOcc/tree/main/projects/configs/baselines) to only generate the Cam4DOcc data without model training and inference.

### Train OCFNetV1.1 with 8 GPUs

OCFNetV1.1 can forecast inflated GMO and others. In this case, _vehicle_ and _human_ are considered as one unified category.

For the nuScenes dataset, please run

```bash
bash run.sh ./projects/configs/baselines/OCFNet_in_Cam4DOcc_V1.1.py 8
```

For the Lyft dataset, please run

```bash
bash run.sh ./projects/configs/baselines/OCFNet_in_Cam4DOcc_V1.1_lyft.py 8
```
### Train OCFNetV1.2 with 8 GPUs

OCFNetV1.2 can forecast inflated GMO including _bicycle_, _bus_, _car_, _construction_, _motorcycle_, _trailer_, _truck_, _pedestrian_, and others. In this case, _vehicle_ and _human_ are divided into multiple categories for clearer evaluation on forecasting performance.

For the nuScenes dataset, please run

```bash
bash run.sh ./projects/configs/baselines/OCFNet_in_Cam4DOcc_V1.2.py 8
```

For the Lyft dataset, please run

```bash
bash run.sh ./projects/configs/baselines/OCFNet_in_Cam4DOcc_V1.2_lyft.py 8
```

* The training/test process will be accelerated several times after you generate datasets by the first epoch.

### Test OCFNet for different tasks

If you only want to test the performance of occupancy prediction for the present frame (current observation), please set `test_present=True` in the [config files](https://github.com/haomo-ai/Cam4DOcc/tree/main/projects/configs/baselines). Otherwise, forecasting performance on the future interval is evaluated.

```bash
bash run_eval.sh $PATH_TO_CFG $PATH_TO_CKPT $GPU_NUM
# e.g. bash run_eval.sh ./projects/configs/baselines/OCFNet_in_Cam4DOcc_V1.1.py ./work_dirs/OCFNet_in_Cam4DOcc_V1.1/epoch_20.pth  8
```
Please set `save_pred` and `save_path` in the config files once saving prediction results is needed.

`VPQ` evaluation of 3D instance prediction will be refined in the future.

### Visualization

Please install the dependencies as follows:

```bash
sudo apt-get install Xvfb
pip install xvfbwrapper
pip install mayavi
```
where `Xvfb` may be needed for visualization in your server.

**Visualize ground-truth occupancy labels**. Set `show_time_change = True` if you want to show the changing state of occupancy in time intervals. 

```bash
cd viz
python viz_gt.py
```
<img src="https://github.com/haomo-ai/Cam4DOcc/blob/main/viz_occupancy.png" width="100%"/>

**Visualize occupancy forecasting results**. Set `show_time_change = True` if you want to show the changing state of occupancy in time intervals. 

```bash
cd viz
python viz_pred.py
```
<img src="https://github.com/haomo-ai/Cam4DOcc/blob/main/viz_pred.png" width="100%"/>

There is still room for improvement. Camera-only 4D occupancy forecasting remains challenging, especially for predicting over longer time intervals with many moving objects. We envision this benchmark as a valuable evaluation tool, and our OCFNet can serve as a foundational codebase for future research on 4D occupancy forecasting.

## Basic Information

Some basic information as well as key parameters for our current version.

| Type |  Info | Parameter |
| :----: | :----: | :----: |
| train           | 23,930 sequences | train_capacity |
| val             | 5,119 frames | test_capacity |
| voxel size      | 0.2m | voxel_x/y/z |
| range           | [-51.2m, -51.2m, -5m, 51.2m, 51.2m, 3m]| point_cloud_range |
| volume size     | [512, 512, 40]| occ_size |
| classes         | 2 for V1.1 / 9 for V1.2 | num_cls |
| observation frames | 3 | time_receptive_field |
| future frames | 4 | n_future_frames |
| extension frames | 6 | n_future_frames_plus |

Our proposed OCFNet can still perform well while being trained with partial data. Please try to decrease `train_capacity` if you want to explore more details with sparser supervision signals. 

In addition, please make sure that `n_future_frames_plus <= time_receptive_field + n_future_frames` because `n_future_frames_plus` means the real prediction number. We estimate more frames including the past ones rather than only `n_future_frames`.

## Pretrained Models

Please download our pretrained models (for epoch=20) to resume training or reproduce results.

| Version | Google Drive <img src="https://ssl.gstatic.com/docs/doclist/images/drive_2022q3_32dp.png" alt="Google Drive" width="18"/> | Baidu Cloud <img src="https://nd-static.bdstatic.com/m-static/v20-main/favicon-main.ico" alt="Baidu Yun" width="18"/> | Config |
| :---: | :---: | :---: | :---: |
| V1.1 | [link](https://drive.google.com/file/d/1IXRqOQk3RKpIjGgBBqV9D9vgSt58QDr8/view?usp=sharing) | [link](https://pan.baidu.com/s/18gODsVnBAXEJ4pzv2-LqGA?pwd=m99b) | [OCFNet_in_Cam4DOcc_V1.1.py](https://github.com/haomo-ai/Cam4DOcc/blob/main/projects/configs/baselines/OCFNet_in_Cam4DOcc_V1.1.py) |
| V1.2 | [link](https://drive.google.com/file/d/1q1XnRt0wYE3oq6YBMBnagpGL7h2I46uN/view?usp=sharing) | [link](https://pan.baidu.com/s/1OPc1-a2McOO_0QPX63J7WQ?pwd=adic) | [OCFNet_in_Cam4DOcc_V1.2.py](https://github.com/haomo-ai/Cam4DOcc/blob/main/projects/configs/baselines/OCFNet_in_Cam4DOcc_V1.2.py) |

## Other Baselines

We also provide the evaluation on the forecasting performance of [other baselines](https://github.com/haomo-ai/Cam4DOcc/tree/main/other_baselines) in Cam4DOcc.

## TODO
The tutorial is being updated ...

We will release our pretrained models as soon as possible. OCFNetV1.3 and OCFNetV2 are on their way ...


### Acknowledgement

We thank the fantastic works [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy), [PowerBEV](https://github.com/EdwardLeeLPZ/PowerBEV), and [FIERY](https://anthonyhu.github.io/fiery) for their pioneer code release, which provide codebase for this benchmark.
