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

We follow the installation instructions of our codebase [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy/blob/main/docs/install.md), which are also posted here:

* Create a conda virtual environment and activate it
```
conda create -n OpenOccupancy python=3.7 -y
conda activate OpenOccupancy
```
* Install PyTorch and torchvision (tested on torch==1.10.1 & cuda=11.3)
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```
* Install gcc>=5 in conda env
```
conda install -c omgarcia gcc-6
```
* Install mmcv, mmdet, and mmseg
```
pip install mmcv-full==1.4.0
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```
* Install mmdet3d from the source code
```
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install
```
* Install other dependencies
```
pip install timm
pip install open3d-python
pip install PyMCubes
pip install spconv-cu113
pip install fvcore
pip install setuptools==59.5.0
```
* Install occupancy pooling
```
git clone git@github.com:haomo-ai/Cam4DOcc.git
cd Cam4DOcc
export PYTHONPATH=“.”
python setup.py develop
```
### Data Structure

Please link your [nuScenes V1.0 full dataset ](https://www.nuscenes.org/nuscenes#download) to the data folder. [nuScenes-Occupancy](https://drive.google.com/file/d/1vTbgddMzUN6nLyWSsCZMb9KwihS7nPoH/view?usp=sharing), [nuscenes_occ_infos_train.pkl](https://github.com/JeffWang987/OpenOccupancy/releases/tag/train_pkl), and [nuscenes_occ_infos_val.pkl](https://github.com/JeffWang987/OpenOccupancy/releases/tag/val_pkl) are also provided by the previous work. If you only want to reproduce the forecasting results with "inflated" form, nuScenes dataset is all you need.

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
│   │   ├── nuscenes_occ_infos_train.pkl/
│   │   ├── nuscenes_occ_infos_val.pkl/
│   ├── nuScenes-Occupancy/
```
## Training and Evalaution

### Train OCFNetV1.1 with 8 GPUs

OCFNetV1.1 can forecast inflated GMO and others. In this case, _vehicle_ and _human_ are considered as one unified category.
```bash
bash run.sh ./projects/configs/baselines/OCFNet_in_Cam4DOcc_V1.1.py 8
```
### Train OCFNetV1.2 with 8 GPUs

OCFNetV1.2 can forecast inflated GMO including _bicycle_, _bus_, _car_, _construction_, _motorcycle_, _trailer_, _truck_, _pedestrian_, and others. In this case, _vehicle_ and _human_ are divided into multiple categories for clearer evaluation on forecasting performance.

```bash
bash run.sh ./projects/configs/baselines/OCFNet_in_Cam4DOcc_V1.2.py 8
```

### Test OCFNet for different tasks

```bash
bash run_eval.sh $PATH_TO_CFG $PATH_TO_CKPT $GPU_NUM
# e.g. bash run_eval.sh ./projects/configs/baselines/OCFNet_in_Cam4DOcc_V1.1.py ./work_dirs/OCFNet_in_Cam4DOcc_V1.1/epoch_20.pth  8
```

## TODO
The tutorial is being refined ...

We will release our pretrained models as soon as possible. OCFNetV1.3 and OCFNetV2 are on their way ...


### Acknowledgement

We thank the fantastic works [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy), [PowerBEV](https://github.com/EdwardLeeLPZ/PowerBEV), and [FIERY](https://anthonyhu.github.io/fiery) for their pioneer code release, which provide codebase for this benchmark.
